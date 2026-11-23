#include "pjrt_runtime.hpp"
#include "stablehlo_emitter.hpp"
#ifdef CLOUDTPU_HAS_GGML
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cloudtpu.h"
#include "ggml-cpu.h"
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
#endif

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

    // Alias passthrough + CompileOptionsProto.env_option_overrides (field 7 map<string, OptionOverrideProto>).
    // Must leave ExecutableBuildOptionsProto.debug_options unset so XLA_FLAGS & defaults stay intact,
    // and serialize each OptionOverrideProto with its exact typed field (bool_field=2, string_field=1, int_field=3).
    cloudtpu::PjrtCompileOptions opt_alias_only;
    opt_alias_only.alias_passthrough_params = true;
    const std::string p_alias_only = cloudtpu::encode_compile_options_proto(opt_alias_only);

    cloudtpu::PjrtCompileOptions opt_extra = opt_alias_only;
    opt_extra.env_option_overrides = {
        cloudtpu::PjrtOptionOverride::make_bool("xla_tpu_detect_nan", true),
        cloudtpu::PjrtOptionOverride::make_string("xla_dump_to", "/tmp/xla_dump"),
        cloudtpu::PjrtOptionOverride::make_int("xla_cpu_parallel_codegen_split_count", 16384),
    };
    const std::string p_extra = cloudtpu::encode_compile_options_proto(opt_extra);
    // Prefix must match p_alias_only byte-for-byte (proves ExecutableBuildOptionsProto is untouched / no debug_options).
    CHECK_TRUE(p_extra.size() > p_alias_only.size() &&
                   std::memcmp(p_extra.data(), p_alias_only.data(), p_alias_only.size()) == 0,
               "env_option_overrides must not modify ExecutableBuildOptionsProto or emit debug_options");
    // Field 7 wire tag = (7<<3)|2 = 0x3a immediately follows ExecutableBuildOptionsProto.
    CHECK_TRUE(static_cast<uint8_t>(p_extra[p_alias_only.size()]) == 0x3a,
               "env_option_overrides must serialize as CompileOptionsProto field 7 (0x3a)");
    CHECK_TRUE(p_extra.find("xla_tpu_detect_nan") != std::string::npos,
               "env_option_overrides bool DebugOptions key missing");
    CHECK_TRUE(p_extra.find("xla_dump_to") != std::string::npos &&
                   p_extra.find("/tmp/xla_dump") != std::string::npos,
               "env_option_overrides string DebugOptions value missing");
    // Verify typed OptionOverrideProto payload bytes:
    //   bool_field=true -> value field (0x12 len=2) + bool_field tag (0x10 0x01)
    const std::string bool_true_wire = {0x12, 0x02, 0x10, 0x01};
    CHECK_TRUE(p_extra.find(bool_true_wire) != std::string::npos,
               "bool override must encode OptionOverrideProto.bool_field (tag 0x10, val 0x01)");
    //   int_field=16384 (0x4000 -> varint 80 80 01) -> value field (0x12 len=4) + int_field tag (0x18 80 80 01)
    const std::string int_16384_wire = {
        0x12, 0x04, 0x18, static_cast<char>(0x80), static_cast<char>(0x80), 0x01};
    CHECK_TRUE(p_extra.find(int_16384_wire) != std::string::npos,
               "int override must encode OptionOverrideProto.int_field (tag 0x18)");
    // Field 10 tag = (10<<3)|0 = 0x50 followed by 0x01 inside ExecutableBuildOptionsProto
    bool found_alias_tag = false;
    for (size_t i = 0; i + 1 < p_alias_only.size(); ++i) {
        if (static_cast<uint8_t>(p_alias_only[i]) == 0x50 && static_cast<uint8_t>(p_alias_only[i + 1]) == 0x01) {
            found_alias_tag = true;
            break;
        }
    }
    CHECK_TRUE(found_alias_tag, "alias_passthrough_params (field 10) missing");

    // Verify LIBTPU_INIT_ARGS helper appends documented async collective flags without clobbering
    // or duplicating user-supplied flags.
    const std::string init_all = cloudtpu::append_async_collective_libtpu_init_args("");
    CHECK_TRUE(init_all.find("--xla_tpu_enable_async_collective_fusion=true") != std::string::npos &&
                   init_all.find("--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true") != std::string::npos &&
                   init_all.find("--xla_tpu_enable_async_collective_fusion_multiple_steps=true") != std::string::npos &&
                   init_all.find("--xla_tpu_overlap_compute_collective_tc=true") != std::string::npos &&
                   init_all.find("--xla_enable_async_all_gather=true") != std::string::npos,
               "append_async_collective_libtpu_init_args must include all 5 documented LIBTPU_INIT_ARGS flags");
    CHECK_TRUE(cloudtpu::append_async_collective_libtpu_init_args(init_all) == init_all,
               "append_async_collective_libtpu_init_args must be idempotent");
    const std::string user_args =
        "--xla_tpu_enable_async_collective_fusion=false --noxla_tpu_overlap_compute_collective_tc "
        "--xla_enable_async_all_gather=kAuto --custom_flag=42";
    const std::string merged = cloudtpu::append_async_collective_libtpu_init_args(user_args);
    CHECK_TRUE(merged.rfind(user_args, 0) == 0, "user LIBTPU_INIT_ARGS prefix must be preserved verbatim");
    CHECK_TRUE(merged.find("--xla_tpu_enable_async_collective_fusion=true") == std::string::npos,
               "must not override user --xla_tpu_enable_async_collective_fusion=false");
    CHECK_TRUE(merged.find("--xla_tpu_overlap_compute_collective_tc=true") == std::string::npos,
               "must not override user --noxla_tpu_overlap_compute_collective_tc");
    CHECK_TRUE(merged.find("--xla_enable_async_all_gather=true") == std::string::npos,
               "must not override user --xla_enable_async_all_gather=kAuto");
    CHECK_TRUE(merged.find("--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true") != std::string::npos &&
                   merged.find("--xla_tpu_enable_async_collective_fusion_multiple_steps=true") != std::string::npos,
               "missing async collective flags must still be appended");
    std::printf("  [PASS] compile_options_proto (incl. DeviceAssignmentProto, alias_passthrough, typed env_option_overrides, LIBTPU_INIT_ARGS)\n");
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

    // Verify get_cached_exec refreshes runtime LRU recency without retaining evicted handles
    // (cap=3: 2 recurring subgraphs + 2 alternating dynamic attention shapes must evict the
    // older attention shape immediately, never the recurring subgraphs).
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

        std::string key_rec0, key_rec1, key_attn0, key_attn1;
        std::weak_ptr<const cloudtpu::CachedExecutable> weak_attn0;
        {
            const auto ex_rec0 = rt3.compile_or_get(p_rec0.cache_key, p_rec0.mlir_text);
            const auto ex_rec1 = rt3.compile_or_get(p_rec1.cache_key, p_rec1.mlir_text);
            const auto ex_attn0 = rt3.compile_or_get(p_attn0.cache_key, p_attn0.mlir_text);
            key_rec0 = ex_rec0->key;
            key_rec1 = ex_rec1->key;
            key_attn0 = ex_attn0->key;
            weak_attn0 = ex_attn0;
        }
        // Simulate decode step plan-cache hits acquiring + refreshing recurring executables in runtime LRU.
        CHECK_TRUE(rt3.get_cached_exec(key_rec0) != nullptr, "get_cached_exec finds rec0");
        CHECK_TRUE(rt3.get_cached_exec(key_rec1) != nullptr, "get_cached_exec finds rec1");
        // Compiling a new attention shape should evict attn0 immediately (weak_attn0 expires!), NOT rec0 or rec1.
        {
            const auto ex_attn1 = rt3.compile_or_get(p_attn1.cache_key, p_attn1.mlir_text);
            key_attn1 = ex_attn1->key;
        }
        CHECK_TRUE(weak_attn0.expired(), "evicted attn0 executable destroyed immediately when unreferenced");
        CHECK_TRUE(rt3.has_cached_exec(key_rec0), "rec0 survives eviction after get_cached_exec");
        CHECK_TRUE(rt3.has_cached_exec(key_rec1), "rec1 survives eviction after get_cached_exec");
        CHECK_TRUE(rt3.has_cached_exec(key_attn1), "attn1 present in cache");
        CHECK_TRUE(!rt3.has_cached_exec(key_attn0), "stale attn0 evicted first");
        CHECK_TRUE(rt3.get_cached_exec(key_attn0) == nullptr, "get_cached_exec returns nullptr for evicted entry");
        // Next step: refresh rec0/rec1 again and verify alternating back to attn0 evicts attn1.
        CHECK_TRUE(rt3.get_cached_exec(key_rec0) != nullptr, "rec0 still hits after attn1 insert");
        CHECK_TRUE(rt3.get_cached_exec(key_rec1) != nullptr, "rec1 still hits after attn1 insert");
        const auto ex_attn0_b = rt3.compile_or_get(p_attn0.cache_key, p_attn0.mlir_text);
        CHECK_TRUE(rt3.has_cached_exec(key_rec0) && rt3.has_cached_exec(key_rec1),
                   "recurring subgraphs never evicted across alternating attention shapes");
        CHECK_TRUE(rt3.has_cached_exec(ex_attn0_b->key) && !rt3.has_cached_exec(key_attn1),
                   "alternating attention shape evicts previous attention shape");
        rt3.shutdown();
    }

    std::printf("  [PASS] H2D/D2H roundtrip & executable LRU cache (incl. get_cached_exec recency)\n");
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

    // 3) Dense F16 weight + F32 activation stays out of the default BF16 quantized fast path:
    //    [1 + 2^-9, 1] . [1, -1] must yield exact 2^-9 (BF16 rounding would collapse both to 1 -> 0).
    {
        const float eps = std::ldexp(1.0f, -9);
        const std::vector<uint16_t> w_f16 = {cloudtpu::f32_to_f16(1.0f + eps), cloudtpu::f32_to_f16(1.0f)};
        const std::vector<float> x_f32 = {1.0f, -1.0f};
        auto bw_f16 = rt.buffer_from_host(0, w_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, 1, 2});
        auto bx_f32 = rt.buffer_from_host(0, x_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 2});

        cloudtpu::GraphProgramSpec f16_spec;
        f16_spec.bf16_matmul = true;  // Default quantized BF16 fast path ON
        f16_spec.inputs.push_back({1, cloudtpu::PjrtDType::F16, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
        f16_spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
        cloudtpu::OpNodeSpec f16_mm;
        f16_mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        f16_mm.src_ids = {1, 2};
        f16_mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}};
        f16_mm.high_precision = false;
        f16_spec.ops.push_back(f16_mm);
        f16_spec.output_ids = {3};

        const auto f16_prog = cloudtpu::emit_stablehlo_program(f16_spec);
        CHECK_TRUE(f16_prog.mlir_text.find("bf16") == std::string::npos,
                   "Dense F16 MUL_MAT must not cast to bf16 under default bf16_matmul=true");
        const auto f16_got = run_emitted(rt, f16_prog, {bw_f16, bx_f32});
        CHECK_TRUE(std::fabs(f16_got[0] - eps) < 1e-7f,
                   "Dense F16 MUL_MAT [1+2^-9, 1] . [1, -1] must equal exact 2^-9");
    }
    std::printf("  [PASS] MUL_MAT (F32 GQA + Q4_0_SPLIT + dense F16 precision)\n");
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
            if (!std::isfinite(nmse) || nmse > 5e-4) {
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
        if (!std::isfinite(nmse) || nmse > 5e-4) {
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
        const auto copts = rt.make_spmd_compile_options(1, 0, /*alias_passthrough=*/false);
        const auto exec = rt.compile_or_get(prog.cache_key + "|alias_test", prog.mlir_text, 0, copts);
        const std::vector<int64_t> non_don = {0, 1};
        std::vector<PJRT_Buffer*> raw_in = {bupd->handle.get(), bidx->handle.get(), btbl->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs,
                   {cloudtpu::PjrtDType::F32}, {{1, 1, R, D}}, true, &non_don);
        std::vector<float> got(static_cast<size_t>(R * D));
        rt.buffer_to_host(outs[0]->handle.get(), got.data(), got.size() * sizeof(float));
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t d = 0; d < D; ++d) {
                const float exp_v = (r == row_idx) ? upd[static_cast<size_t>(d)] : tbl[static_cast<size_t>(r * D + d)];
                CHECK_NEAR(got[static_cast<size_t>(r * D + d)], exp_v, 1e-5f, "SET_ROWS DUS KV update");
            }
        }

        // Out-of-range indices (-1 and R) must leave every KV row bit-for-bit untouched (matching scatter).
        // Upload a fresh KV buffer for each case because the preceding execution allows PJRT buffer donation.
        for (int32_t bad_idx : {int32_t{-1}, static_cast<int32_t>(R)}) {
            auto bbad = rt.buffer_from_host(0, &bad_idx, cloudtpu::PjrtDType::S32, {1, 1, 1, 1});
            auto btbl_case = rt.buffer_from_host(0, tbl.data(), cloudtpu::PjrtDType::F32, {1, 1, R, D});
            std::vector<PJRT_Buffer*> bad_in = {bupd->handle.get(), bbad->handle.get(), btbl_case->handle.get()};
            std::vector<cloudtpu::PjrtBufferPtr> bad_outs;
            rt.execute(exec, 0, bad_in.data(), bad_in.size(), bad_outs,
                       {cloudtpu::PjrtDType::F32}, {{1, 1, R, D}}, true, &non_don);
            std::vector<float> bad_got(static_cast<size_t>(R * D));
            rt.buffer_to_host(bad_outs[0]->handle.get(), bad_got.data(), bad_got.size() * sizeof(float));
            check_vec_close(bad_got, tbl, 0.0f, "SET_ROWS DUS out-of-range preserves KV table");
        }
    }

    std::printf("  [PASS] Silicon regressions (ROPE M=52, Q4_0/Q8_0 K=256, bounds-guarded DUS+KV donation)\n");
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

    // Verify BF16 flash attention is gated independently of bf16_matmul so default
    // CLOUDTPU_MATMUL_BF16=1 preserves F32 GGML attention parity (<0.0005 NMSE).
    {
        constexpr int64_t D256 = 256;
        std::vector<float> q256(static_cast<size_t>(D256));
        std::vector<float> k256(static_cast<size_t>(2 * D256));
        std::vector<float> v256(static_cast<size_t>(2 * D256), 0.0f);
        for (int64_t i = 0; i < 128; ++i) {
            q256[static_cast<size_t>(i)] = 1.00390625f;  // 1 + 2^-8 (distinct from 1 in F32, rounds to 1 in BF16)
            q256[static_cast<size_t>(128 + i)] = 1.0f;
            k256[static_cast<size_t>(i)] = 2.0f;
            k256[static_cast<size_t>(128 + i)] = -2.0f;
            k256[static_cast<size_t>(D256 + i)] = -2.0f;
            k256[static_cast<size_t>(D256 + 128 + i)] = 2.0f;
        }
        v256[0] = 1.0f;
        v256[static_cast<size_t>(D256)] = 0.0f;
        auto bq256 = rt.buffer_from_host(0, q256.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, D256});
        auto bk256 = rt.buffer_from_host(0, k256.data(), cloudtpu::PjrtDType::F32, {1, 1, 2, D256});
        auto bv256 = rt.buffer_from_host(0, v256.data(), cloudtpu::PjrtDType::F32, {1, 1, 2, D256});

        cloudtpu::GraphProgramSpec fa_spec;
        fa_spec.bf16_matmul = true;       // Default matmul BF16 ON
        fa_spec.bf16_flash_attn = false;  // Default flash attention BF16 OFF
        fa_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D256}});
        fa_spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, D256}});
        fa_spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, D256}});
        cloudtpu::OpNodeSpec fa_op;
        fa_op.kind = cloudtpu::CloudOpKind::FLASH_ATTN_EXT;
        fa_op.src_ids = {1, 2, 3};
        fa_op.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D256}};
        fa_op.flash_attn.scale = 1.0f / 16.0f;
        fa_op.high_precision = false;
        fa_spec.ops.push_back(fa_op);
        fa_spec.output_ids = {4};

        const auto fa_f32_prog = cloudtpu::emit_stablehlo_program(fa_spec);
        CHECK_TRUE(fa_f32_prog.mlir_text.find("bf16") == std::string::npos,
                   "Default bf16_flash_attn=false must keep FLASH_ATTN_EXT in F32 even when bf16_matmul=true");
        const auto fa_f32_out = run_emitted(rt, fa_f32_prog, {bq256, bk256, bv256});
        const float exp_f32 = std::exp(0.0625f) / (std::exp(0.0625f) + std::exp(-0.0625f));
        CHECK_TRUE(std::fabs(fa_f32_out[0] - exp_f32) < 1e-5f,
                   "Default FLASH_ATTN_EXT must match F32 reference within GGML parity tolerance");

        fa_spec.bf16_flash_attn = true;
        const auto fa_bf16_prog = cloudtpu::emit_stablehlo_program(fa_spec);
        CHECK_TRUE(fa_bf16_prog.mlir_text.find("bf16") != std::string::npos,
                   "Opt-in bf16_flash_attn=true must cast FLASH_ATTN_EXT operands to bf16");
        CHECK_TRUE(fa_bf16_prog.cache_key != fa_f32_prog.cache_key,
                   "bf16_flash_attn toggle must produce distinct executable cache keys");
        const auto fa_bf16_out = run_emitted(rt, fa_bf16_prog, {bq256, bk256, bv256});
        CHECK_TRUE(std::fabs(fa_bf16_out[0] - 0.5f) < 1e-5f,
                   "Opt-in BF16 FLASH_ATTN_EXT should reflect BF16 rounding (0.5f)");
    }
    std::printf("  [PASS] FLASH_ATTN_EXT isolates masked NaN/Inf K/V rows & gates BF16 independently\n");
}

void test_q4_k_split_mul_mat(cloudtpu::PjrtRuntime& rt) {
    // Verify Q4_K_SPLIT emission & execution preserves rank-4 K-major dims for both scales (d) and mins (m).
    constexpr int64_t M = 2, K = 32, N = 3, NB = K / 32;
    std::vector<uint8_t> qs_km(static_cast<size_t>(NB * 16 * N), 0);
    std::vector<float> d_km(static_cast<size_t>(NB * N), 0.5f);
    std::vector<float> m_km(static_cast<size_t>(NB * N), -1.0f);
    std::vector<float> w_ref(static_cast<size_t>(N * K), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        d_km[static_cast<size_t>(n)] = 0.25f * static_cast<float>(n + 1);
        m_km[static_cast<size_t>(n)] = -0.5f * static_cast<float>(n + 1);
        for (int64_t i = 0; i < 16; ++i) {
            const uint8_t lo = static_cast<uint8_t>((n + i) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>((n + i + 3) & 0x0F);
            qs_km[static_cast<size_t>(i * N + n)] = static_cast<uint8_t>(lo | (hi << 4));
            w_ref[static_cast<size_t>(n * K + i)] = static_cast<float>(lo) * d_km[static_cast<size_t>(n)] + m_km[static_cast<size_t>(n)];
            w_ref[static_cast<size_t>(n * K + 16 + i)] = static_cast<float>(hi) * d_km[static_cast<size_t>(n)] + m_km[static_cast<size_t>(n)];
        }
    }
    std::vector<float> x(static_cast<size_t>(M * K));
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * static_cast<float>(static_cast<int>(i % 7) - 3);

    auto bqs = rt.buffer_from_host(0, qs_km.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 16, N});
    auto bd = rt.buffer_from_host(0, d_km.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
    auto bm = rt.buffer_from_host(0, m_km.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
    auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

    // 1) Strict F32 parity path (bf16_matmul=false, high_precision=true)
    cloudtpu::GraphProgramSpec spec;
    spec.bf16_matmul = false;
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
    CHECK_TRUE(prog.input_bindings[2].dims == std::vector<int64_t>({1, 1, NB, N}), "Q4_K mins binding rank-4 K-major");
    CHECK_TRUE(prog.mlir_text.find("bf16") == std::string::npos, "Strict F32 Q4_K MUL_MAT must not cast to bf16");

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
    check_vec_close(got, expected, 1e-4f, "Q4_K_SPLIT MUL_MAT F32");

    // 2) BF16 MXU contraction path (bf16_matmul=true, high_precision=false) verified against
    //    BF16-rounded operands (matching real MXU BF16 input precision).
    cloudtpu::GraphProgramSpec bf16_spec = spec;
    bf16_spec.bf16_matmul = true;
    bf16_spec.ops[0].high_precision = false;
    const auto bf16_prog = cloudtpu::emit_stablehlo_program(bf16_spec);
    CHECK_TRUE(bf16_prog.mlir_text.find("bf16") != std::string::npos,
               "BF16 Q4_K MUL_MAT must cast dequantized weights and activations to bf16");
    const auto got_bf16 = run_emitted(rt, bf16_prog, {bqs, bd, bm, bx});
    std::vector<float> expected_bf16(static_cast<size_t>(M * N), 0.0f);
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float x_q = cloudtpu::bf16_to_f32(cloudtpu::f32_to_bf16(x[static_cast<size_t>(r * K + k)]));
                const float w_q = cloudtpu::bf16_to_f32(cloudtpu::f32_to_bf16(w_ref[static_cast<size_t>(n * K + k)]));
                acc += x_q * w_q;
            }
            expected_bf16[static_cast<size_t>(r * N + n)] = acc;
        }
    }
    check_vec_close(got_bf16, expected_bf16, 1e-4f, "Q4_K_SPLIT MUL_MAT BF16");

    // 3) Compact BF16 scale storage (CLOUDTPU_Q4K_BF16_SCALES=1) preserves finite effective scales
    //    above F16's 65504 limit (e.g. Q6_K d=1024 * sc=64 = 65536) without overflowing to Inf/NaN.
    {
        const float big_scale = 1024.0f * 64.0f;  // 65536.0f (overflows F16 to +Inf, exact in BF16)
        const uint16_t big_f16 = cloudtpu::f32_to_f16(big_scale);
        const uint16_t big_bf16 = cloudtpu::f32_to_bf16(big_scale);
        CHECK_TRUE(std::isinf(cloudtpu::f16_to_f32(big_f16)), "sanity: 65536 overflows F16 to Inf");
        CHECK_TRUE(std::isfinite(cloudtpu::bf16_to_f32(big_bf16)) &&
                       cloudtpu::bf16_to_f32(big_bf16) == big_scale,
                   "BF16 compact scale must preserve 65536.0f exactly");

        constexpr int64_t K6 = 16, N6 = 2, NB6 = K6 / 16;
        std::vector<int8_t> qs6(static_cast<size_t>(NB6 * 16 * N6), 0);
        // Row 0 has all-zero coefficients (0 * Inf would become NaN under F16); row 1 has q[0]=1.
        qs6[static_cast<size_t>(0 * N6 + 1)] = 1;
        std::vector<uint16_t> d6_bf16(static_cast<size_t>(NB6 * N6), big_bf16);
        std::vector<float> x6(static_cast<size_t>(K6), 0.0f);
        x6[0] = 0.5f;

        auto bqs6 = rt.buffer_from_host(0, qs6.data(), cloudtpu::PjrtDType::S8, {1, 1, NB6, 16, N6});
        auto bd6 = rt.buffer_from_host(0, d6_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB6, N6});
        auto bx6 = rt.buffer_from_host(0, x6.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, K6});

        cloudtpu::GraphProgramSpec q6_spec;
        q6_spec.bf16_matmul = false;
        cloudtpu::TensorSpec w6_ts = {
            1, cloudtpu::PjrtDType::S8, cloudtpu::TensorStorageKind::Q6_K_SPLIT, {1, 1, N6, K6}};
        w6_ts.scale_dtype = cloudtpu::PjrtDType::BF16;
        q6_spec.inputs.push_back(w6_ts);
        q6_spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, K6}});
        cloudtpu::OpNodeSpec q6_mm;
        q6_mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        q6_mm.src_ids = {1, 2};
        q6_mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, N6}};
        q6_mm.high_precision = true;
        q6_spec.ops.push_back(q6_mm);
        q6_spec.output_ids = {3};

        const auto q6_prog = cloudtpu::emit_stablehlo_program(q6_spec);
        CHECK_TRUE(q6_prog.input_bindings[1].dtype == cloudtpu::PjrtDType::BF16,
                   "Q6_K_SPLIT compact scale binding must use BF16");
        const auto q6_out = run_emitted(rt, q6_prog, {bqs6, bd6, bx6});
        CHECK_TRUE(std::isfinite(q6_out[0]) && q6_out[0] == 0.0f,
                   "Q6_K_SPLIT BF16 scale=65536 with q=0 must stay finite 0.0 (no NaN)");
        CHECK_TRUE(std::isfinite(q6_out[1]) && std::fabs(q6_out[1] - 32768.0f) < 1e-3f,
                   "Q6_K_SPLIT BF16 scale=65536 with q=1, x=0.5 must equal 32768.0");
    }
    std::printf("  [PASS] Q4_K_SPLIT & Q6_K_SPLIT MUL_MAT (F32 + BF16 MXU + BF16 compact scales)\n");

    // Verify SET_ROWS preserves 64-bit indices on both single-row DUS and multi-row scatter paths.
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
    CHECK_TRUE(sr_prog.mlir_text.find("stablehlo.dynamic_update_slice") != std::string::npos &&
                   sr_prog.mlir_text.find("tensor<i64>") != std::string::npos,
               "Single-row SET_ROWS with S64 indices must preserve i64 dynamic_update_slice coordinates");
    const auto sr_out = run_emitted(rt, sr_prog, {b_upd, b_idx, b_tbl});
    check_vec_close(sr_out, {1.0f, 2.0f, 9.0f, 8.0f, 5.0f, 6.0f}, 1e-6f, "SET_ROWS S64 DUS");
    // Out-of-range S64 indices (-1, 3, and 2^33 which would wrap to 0 if narrowed to i32) must
    // preserve tbl_init unchanged.
    for (int64_t bad64 : {int64_t{-1}, int64_t{3}, int64_t{1LL << 33}}) {
        auto b_bad64 = rt.buffer_from_host(0, &bad64, cloudtpu::PjrtDType::S64, {1, 1, 1, 1});
        const auto bad_out = run_emitted(rt, sr_prog, {b_upd, b_bad64, b_tbl});
        check_vec_close(bad_out, tbl_init, 0.0f, "SET_ROWS S64 DUS out-of-range preserves table");
    }

    // Multi-row SET_ROWS with S64 indices uses scatter and preserves i64 coordinates.
    const std::vector<float> upd_vals2 = {9.0f, 8.0f, 7.0f, 6.5f};
    const std::vector<int64_t> row_idx64_2 = {2, 0};
    auto b_upd2 = rt.buffer_from_host(0, upd_vals2.data(), cloudtpu::PjrtDType::F32, {1, 1, 2, 2});
    auto b_idx2 = rt.buffer_from_host(0, row_idx64_2.data(), cloudtpu::PjrtDType::S64, {1, 1, 1, 2});
    cloudtpu::GraphProgramSpec sr2_spec;
    sr2_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 2}});
    sr2_spec.inputs.push_back({2, cloudtpu::PjrtDType::S64, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
    sr2_spec.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 3, 2}});
    sr2_spec.ops.push_back(sr_op);
    sr2_spec.output_ids = {4};
    const auto sr2_prog = cloudtpu::emit_stablehlo_program(sr2_spec);
    CHECK_TRUE(sr2_prog.mlir_text.find("tensor<1x1x2x3xi64>") != std::string::npos,
               "Multi-row SET_ROWS with S64 indices must preserve i64 scatter coordinates");
    const auto sr2_out = run_emitted(rt, sr2_prog, {b_upd2, b_idx2, b_tbl});
    check_vec_close(sr2_out, {7.0f, 6.5f, 3.0f, 4.0f, 9.0f, 8.0f}, 1e-6f, "SET_ROWS S64 scatter");
    std::printf("  [PASS] SET_ROWS preserves 64-bit indices (DUS + scatter)\n");

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
    CHECK_TRUE(copts.env_option_overrides.empty(),
               "SPMD compile options must not emit non-DebugOptions xla_tpu_* env_option_overrides");

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

        // Quantized Q8_0_SPLIT SHARD_N table (K=32, 1 block per row, K-major N-minor per chip: [1, 1, 1, 32, N_LOCAL])
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
            std::vector<int8_t> qs_local_km(static_cast<size_t>(K * N_LOCAL));
            for (int64_t r = 0; r < N_LOCAL; ++r) {
                for (int64_t c = 0; c < K; ++c) {
                    qs_local_km[static_cast<size_t>(c * N_LOCAL + r)] =
                        qs[static_cast<size_t>((p * N_LOCAL + r) * K + c)];
                }
            }
            auto bq = rt.buffer_from_host(
                p, qs_local_km.data(), cloudtpu::PjrtDType::S8, {1, 1, 1, K, N_LOCAL});
            auto bd = rt.buffer_from_host(
                p, ds.data() + static_cast<size_t>(p * N_LOCAL), cloudtpu::PjrtDType::F16, {1, 1, 1, N_LOCAL});
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

        // 4) Bounds checks: invalid collective_dim must throw std::invalid_argument, and
        //    MockPJRT must reject input shape/dtype mismatches before evaluating collectives.
        for (int64_t bad_dim : std::vector<int64_t>{-1, 4}) {
            bool ag_threw = false;
            try {
                cloudtpu::GraphProgramSpec bad;
                bad.spmd_chips = P;
                bad.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 4}});
                cloudtpu::OpNodeSpec op;
                op.kind = cloudtpu::CloudOpKind::ALL_GATHER;
                op.collective_dim = bad_dim;
                op.src_ids = {1};
                op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 8}};
                bad.ops.push_back(op);
                bad.output_ids = {2};
                (void)cloudtpu::emit_stablehlo_program(bad);
            } catch (const std::invalid_argument&) {
                ag_threw = true;
            }
            CHECK_TRUE(ag_threw, "ALL_GATHER with out-of-bounds collective_dim must throw invalid_argument");

            bool rs_threw = false;
            try {
                cloudtpu::GraphProgramSpec bad;
                bad.spmd_chips = P;
                bad.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 4}});
                cloudtpu::OpNodeSpec op;
                op.kind = cloudtpu::CloudOpKind::REDUCE_SCATTER_SUM;
                op.collective_dim = bad_dim;
                op.src_ids = {1};
                op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 2, 2}};
                bad.ops.push_back(op);
                bad.output_ids = {2};
                (void)cloudtpu::emit_stablehlo_program(bad);
            } catch (const std::invalid_argument&) {
                rs_threw = true;
            }
            CHECK_TRUE(rs_threw, "REDUCE_SCATTER_SUM with out-of-bounds collective_dim must throw invalid_argument");
        }
        {
            cloudtpu::GraphProgramSpec ag_spec;
            ag_spec.spmd_chips = P;
            ag_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 2}});
            cloudtpu::OpNodeSpec op;
            op.kind = cloudtpu::CloudOpKind::ALL_GATHER;
            op.collective_dim = 3;
            op.src_ids = {1};
            op.dst = {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 4}};
            ag_spec.ops.push_back(op);
            ag_spec.output_ids = {2};
            const auto ag_prog = cloudtpu::emit_stablehlo_program(ag_spec);
            const auto ag_exec = rt.compile_or_get(ag_prog.cache_key, ag_prog.mlir_text, 0, copts);

            // Shape mismatch: pass [1, 1, 1, 4] instead of expected [1, 1, 1, 2]
            const std::vector<float> full_vals = {1.0f, 2.0f, 3.0f, 4.0f};
            auto wrong_s0 = rt.buffer_from_host(0, full_vals.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
            auto wrong_s1 = rt.buffer_from_host(1, full_vals.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4});
            PJRT_Buffer* ws0[1] = {wrong_s0->handle.get()};
            PJRT_Buffer* ws1[1] = {wrong_s1->handle.get()};
            PJRT_Buffer* const* ws_ptrs[2] = {ws0, ws1};
            bool shape_rejected = false;
            try {
                std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
                rt.execute_spmd(ag_exec, P, ws_ptrs, 1, outs, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 4}}, true);
            } catch (const std::exception&) {
                shape_rejected = true;
            }
            CHECK_TRUE(shape_rejected, "MockPJRT must reject mismatched SPMD input shape");

            // Dtype mismatch: pass S32 [1, 1, 1, 2] instead of expected F32 [1, 1, 1, 2]
            const std::vector<int32_t> wrong_dt_vals = {1, 2};
            auto wd0 = rt.buffer_from_host(0, wrong_dt_vals.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 2});
            auto wd1 = rt.buffer_from_host(1, wrong_dt_vals.data(), cloudtpu::PjrtDType::S32, {1, 1, 1, 2});
            PJRT_Buffer* d0[1] = {wd0->handle.get()};
            PJRT_Buffer* d1[1] = {wd1->handle.get()};
            PJRT_Buffer* const* wd_ptrs[2] = {d0, d1};
            bool dtype_rejected = false;
            try {
                std::vector<std::vector<cloudtpu::PjrtBufferPtr>> outs;
                rt.execute_spmd(ag_exec, P, wd_ptrs, 1, outs, {cloudtpu::PjrtDType::F32}, {{1, 1, 1, 4}}, true);
            } catch (const std::exception&) {
                dtype_rejected = true;
            }
            CHECK_TRUE(dtype_rejected, "MockPJRT must reject mismatched SPMD input dtype");
        }

        // 5) PendingSyncLaunchQueue: a single 8-chip 3-output (24-event) launch must stay queued
        //    without forcing a wait on itself, and exceeding the launch cap drains only the oldest launch.
        {
            cloudtpu::RuntimeConfig cfg8;
            cfg8.mock_mode = true;
            cfg8.mock_devices = 8;
            cfg8.exec_cache_cap = 16;
            cloudtpu::PjrtRuntime rt8;
            rt8.init(cfg8);

            const std::vector<float> dummy(4, 1.0f);
            std::vector<cloudtpu::PjrtBufferPtr> chip_bufs;
            chip_bufs.reserve(8);
            for (int d = 0; d < 8; ++d) {
                chip_bufs.push_back(rt8.buffer_from_host(d, dummy.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, 4}));
            }

            auto make_24_event_launch = [&]() -> std::vector<cloudtpu::PjrtEventHandle> {
                std::vector<cloudtpu::PjrtEventHandle> evs;
                evs.reserve(24);
                for (int out = 0; out < 3; ++out) {
                    for (int d = 0; d < 8; ++d) {
                        auto ev = rt8.create_buffer_ready_event(chip_bufs[static_cast<size_t>(d)]->handle.get());
                        if (ev) {
                            evs.push_back(std::move(ev));
                        }
                    }
                }
                return evs;
            };

            cloudtpu::PendingSyncLaunchQueue q;
            constexpr size_t kCap = cloudtpu::PendingSyncLaunchQueue::kDefaultMaxLaunches;
            auto first_launch = make_24_event_launch();
            CHECK_TRUE(first_launch.size() == 24, "8-chip 3-output launch created 24 valid events");
            auto ov0 = q.enqueue(std::move(first_launch), kCap);
            CHECK_TRUE(ov0.empty(), "Single 24-event 8-chip launch must not trigger self-wait overflow");
            CHECK_TRUE(q.pending_launches() == 1, "Single 24-event launch counts as 1 pending launch");

            for (size_t k = 1; k < kCap; ++k) {
                auto ov = q.enqueue(make_24_event_launch(), kCap);
                CHECK_TRUE(ov.empty(), "Up to kCap launches must stay queued without overflow");
            }
            CHECK_TRUE(q.pending_launches() == kCap, "Queue holds exactly kCap launches");

            auto ov_exceed = q.enqueue(make_24_event_launch(), kCap);
            CHECK_TRUE(ov_exceed.size() == 1 && ov_exceed.front().size() == 24,
                       "Exceeding launch cap drains exactly 1 oldest 24-event launch");
            CHECK_TRUE(q.pending_launches() == kCap, "Queue stays capped at kCap launches after overflow");
            for (auto& ev : ov_exceed.front()) {
                rt8.await_event(std::move(ev));
            }

            auto remaining = q.drain_all();
            CHECK_TRUE(remaining.size() == kCap && q.pending_launches() == 0,
                       "drain_all returns all remaining launches and empties queue");
            for (auto& launch : remaining) {
                CHECK_TRUE(launch.size() == 24, "Each drained launch has all 24 events intact");
                for (auto& ev : launch) {
                    rt8.await_event(std::move(ev));
                }
            }
            chip_bufs.clear();
            rt8.shutdown();
        }

        rt.shutdown();
        std::printf("  [PASS] SPMD REDUCE_SCATTER_SUM -> CPY, S32 precision, bounds checks, & per-launch async queue\n");
    }
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

        // Reading an S4 tensor (17 elements incl. -8, -1) into an oversized 32-byte host buffer
        // must sign-extend exactly nelems=17 bytes and leave spare trailing bytes untouched.
        auto bs4_odd = rt.buffer_from_host(0, s4_src.data(), cloudtpu::PjrtDType::S4, {1, 1, 1, 17});
        CHECK_TRUE(bs4_odd->num_bytes() == 9, "odd S4 num_bytes == 9");
        const int8_t sentinel = static_cast<int8_t>(0x5A);
        std::vector<int8_t> s4_oversized(32, sentinel);
        rt.buffer_to_host(bs4_odd->handle.get(), s4_oversized.data(), s4_oversized.size());
        for (size_t i = 0; i < s4_src.size(); ++i) {
            CHECK_TRUE(s4_oversized[i] == s4_src[i],
                       "S4 H2D/D2H into oversized buffer must sign-extend all nelems");
        }
        for (size_t i = s4_src.size(); i < s4_oversized.size(); ++i) {
            CHECK_TRUE(s4_oversized[i] == sentinel,
                       "S4 H2D/D2H into oversized buffer must leave spare trailing bytes untouched");
        }

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

    // 2) Q4_K_I4: S4 (unshifted m) with F32 & F16 scales, plus U4 (0..15) & bitcast
    {
        const int64_t N = 16, NB = 8, K = NB * 32;  // K=256 (1 QK_K block)
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint8_t> qs_u4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint8_t> qs_u8_bc(static_cast<size_t>(NB * 32 * (N / 2)));
        std::vector<float> d_f32(static_cast<size_t>(NB * N));
        std::vector<float> m_orig_f32(static_cast<size_t>(NB * N));
        std::vector<uint16_t> d_f16(static_cast<size_t>(NB * N));
        std::vector<uint16_t> m_orig_f16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float d = 0.03f * static_cast<float>((n * 2 + b) % 9 + 1);
                const float m = -0.15f + 0.025f * static_cast<float>((n + b * 3) % 11);
                const size_t bn = static_cast<size_t>(b * N + n);
                d_f32[bn] = d;
                m_orig_f32[bn] = m;
                d_f16[bn] = cloudtpu::f32_to_f16(d);
                m_orig_f16[bn] = cloudtpu::f32_to_f16(m);
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
        auto bm_orig_f32 = rt.buffer_from_host(0, m_orig_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bd_f16 = rt.buffer_from_host(0, d_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});
        auto bm_orig_f16 = rt.buffer_from_host(0, m_orig_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});

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
                auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bqs_s4, bd_f32, bm_orig_f32, bx});
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
                auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(spec), {bqs_s4, bd_f16, bm_orig_f16, bx});
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
                auto got_bc = run_emitted(rt, bc_prog, {bqs_bc, bd_f32, bm_orig_f32, bx});
                CHECK_TRUE(compute_nmse(got_bc, ref) < 5e-4, "Q4_K_I4 bitcast NMSE < 5e-4");
            }
        }
    }

    // 3) Q4_K_I4 exact unshifted minimum & zero-nibble cancellation regressions:
    //    (a) Small-batch MUL_MAT with d = 0.06256103515625, m = 0, u4 = 0 (s4 = -8), x = 1.0 -> exact 0.0f
    //    (b) Small-batch MUL_MAT + GET_ROWS (S4 & U8 bitcast) with huge d = 65504*63, m = -1, u4 = 0 (s4 = -8)
    //        -> exact -1.0f (would round m + 8*d to 8*d and return 0.0f if m were pre-shifted!)
    {
        const int64_t N = 4, NB = 8, K = NB * 32, M = 1;
        struct ExactCase {
            float d_val;
            float m_val;
            float expected;
        };
        const ExactCase cases[] = {
            {0.06256103515625f, 0.0f, 0.0f},
            {65504.0f * 63.0f, -1.0f, -1.0f},
        };
        for (const ExactCase& ec : cases) {
            std::vector<int8_t> qs_zero(static_cast<size_t>(NB * 32 * N), static_cast<int8_t>(-8));
            std::vector<uint8_t> qs_zero_bc(static_cast<size_t>(NB * 32 * (N / 2)), 0x88U);  // lo=-8, hi=-8
            std::vector<float> d_vec(static_cast<size_t>(NB * N), ec.d_val);
            std::vector<float> m_vec(static_cast<size_t>(NB * N), ec.m_val);
            std::vector<float> x_unit(static_cast<size_t>(M * K), 0.0f);
            x_unit[0] = 1.0f;

            auto bqs = rt.buffer_from_host(0, qs_zero.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
            auto bqs_bc = rt.buffer_from_host(0, qs_zero_bc.data(), cloudtpu::PjrtDType::U8, {1, 1, NB, 32, N / 2});
            auto bd = rt.buffer_from_host(0, d_vec.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bm = rt.buffer_from_host(0, m_vec.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bx = rt.buffer_from_host(0, x_unit.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

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

            const auto prog = cloudtpu::emit_stablehlo_program(spec);
            const size_t first_dot = prog.mlir_text.find("stablehlo.dot_general");
            CHECK_TRUE(first_dot != std::string::npos &&
                           prog.mlir_text.find("stablehlo.dot_general", first_dot + 1) == std::string::npos,
                       "Q4_K_I4 factored path must apply per-block xsum*m before block reduction (no second dot_general)");
            const auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
            CHECK_TRUE(got.size() == static_cast<size_t>(M * N), "Q4_K_I4 exact-m output size");
            for (size_t i = 0; i < got.size(); ++i) {
                CHECK_TRUE(got[i] == ec.expected,
                           "Q4_K_I4 F32-scale zero-nibble MUL_MAT must preserve exact unshifted m");
            }

            // GET_ROWS lookup on both direct S4 and U8 bitcast must also preserve exact unshifted m
            const int32_t row0 = 1;
            auto bidx = rt.buffer_from_host(0, &row0, cloudtpu::PjrtDType::S32, {1, 1, 1, 1});
            for (bool use_bc : {false, true}) {
                cloudtpu::GraphProgramSpec gr_spec;
                cloudtpu::TensorSpec tbl_spec{
                    1, use_bc ? cloudtpu::PjrtDType::U8 : cloudtpu::PjrtDType::S4,
                    cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
                tbl_spec.scale_dtype = cloudtpu::PjrtDType::F32;
                tbl_spec.bitcast_target = cloudtpu::PjrtDType::S4;
                gr_spec.inputs.push_back(tbl_spec);
                gr_spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
                cloudtpu::OpNodeSpec gr;
                gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
                gr.src_ids = {1, 2};
                gr.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, K}};
                gr_spec.ops.push_back(gr);
                gr_spec.output_ids = {3};
                const auto gr_got = run_emitted(rt, cloudtpu::emit_stablehlo_program(gr_spec),
                                                {use_bc ? bqs_bc : bqs, bd, bm, bidx});
                CHECK_TRUE(gr_got.size() == static_cast<size_t>(K), "Q4_K_I4 GET_ROWS exact-m size");
                for (size_t k = 0; k < gr_got.size(); ++k) {
                    CHECK_TRUE(gr_got[k] == ec.expected,
                               "Q4_K_I4 GET_ROWS zero-nibble must preserve exact unshifted m even when 8*d >> |m|");
                }
            }
        }

        // (c) Activation precision parity between dot_b and 8*xsum for zero unsigned nibbles (s4=-8)
        //     with x = 1 + 2^-9 (rounds to 1.0 in BF16) under both bf16_matmul=false and bf16_matmul=true:
        {
            std::vector<int8_t> qs_zero(static_cast<size_t>(NB * 32 * N), static_cast<int8_t>(-8));
            std::vector<float> d_one(static_cast<size_t>(NB * N), 1.0f);
            std::vector<float> m_zero(static_cast<size_t>(NB * N), 0.0f);
            std::vector<float> x_sub_bf16(static_cast<size_t>(M * K), 0.0f);
            for (int64_t k = 0; k < 32; ++k) {
                x_sub_bf16[static_cast<size_t>(k)] = 1.0f + std::ldexp(1.0f, -9);
            }
            auto bqs = rt.buffer_from_host(0, qs_zero.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
            auto bd = rt.buffer_from_host(0, d_one.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bm = rt.buffer_from_host(0, m_zero.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bx = rt.buffer_from_host(0, x_sub_bf16.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

            for (bool bf16_mm : {false, true}) {
                cloudtpu::GraphProgramSpec pspec;
                pspec.bf16_matmul = bf16_mm;
                cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
                w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
                pspec.inputs.push_back(w_spec);
                pspec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.high_precision = false;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                pspec.ops.push_back(mm);
                pspec.output_ids = {3};

                const auto prog = cloudtpu::emit_stablehlo_program(pspec);
                if (!bf16_mm) {
                    CHECK_TRUE(prog.mlir_text.find("precision = [DEFAULT, DEFAULT]") == std::string::npos,
                               "Q4_K_I4 F32 path (CLOUDTPU_MATMUL_BF16=0) must emit HIGHEST precision on dot_b");
                    CHECK_TRUE(prog.mlir_text.find("precision = [HIGHEST, HIGHEST]") != std::string::npos,
                               "Q4_K_I4 F32 path must use HIGHEST precision for dot_b");
                } else {
                    CHECK_TRUE(
                        prog.mlir_text.find("(tensor<1x1x8x8x32xbf16>) -> tensor<1x1x8x8x32xf32>") != std::string::npos,
                        "Q4_K_I4 BF16 path must widen BF16-rounded x_blk to F32 before reducing xsum");
                }
                const auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
                CHECK_TRUE(got.size() == static_cast<size_t>(M * N), "Q4_K_I4 zero-nibble precision parity size");
                for (float v : got) {
                    CHECK_TRUE(v == 0.0f,
                               "Q4_K_I4 zero-nibble dot_b and 8*xsum must cancel identically to 0.0f");
                }
            }
        }

        // (d) Cross-block cancellation regression:
        //     Block 0 has 32 unit activations, u4=1 (s4=-7), d=2^20, m=-2^20 -> net block value 0.
        //     Block 1 has 32 unit activations, u4=1 (s4=-7), d=1/32, m=0     -> net block value 1.0f.
        //     Remaining blocks have d=0, m=0. Reducing d*dot_u4 before adding xsum*m would round
        //     2^25 + 1 -> 2^25 in F32 and return 0.0f; per-block offset addition preserves exact 1.0f!
        {
            std::vector<int8_t> qs_one(static_cast<size_t>(NB * 32 * N), static_cast<int8_t>(-7));  // u4 = 1
            std::vector<float> d_blk(static_cast<size_t>(NB * N), 0.0f);
            std::vector<float> m_blk(static_cast<size_t>(NB * N), 0.0f);
            for (int64_t n = 0; n < N; ++n) {
                d_blk[static_cast<size_t>(0 * N + n)] = std::ldexp(1.0f, 20);
                m_blk[static_cast<size_t>(0 * N + n)] = -std::ldexp(1.0f, 20);
                d_blk[static_cast<size_t>(1 * N + n)] = 1.0f / 32.0f;
                m_blk[static_cast<size_t>(1 * N + n)] = 0.0f;
            }
            std::vector<float> x_ones(static_cast<size_t>(M * K), 0.0f);
            for (int64_t k = 0; k < 64; ++k) {
                x_ones[static_cast<size_t>(k)] = 1.0f;
            }
            auto bqs = rt.buffer_from_host(0, qs_one.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
            auto bd = rt.buffer_from_host(0, d_blk.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bm = rt.buffer_from_host(0, m_blk.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
            auto bx = rt.buffer_from_host(0, x_ones.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

            for (bool bf16_mm : {false, true}) {
                cloudtpu::GraphProgramSpec cspec;
                cspec.bf16_matmul = bf16_mm;
                cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
                w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
                cspec.inputs.push_back(w_spec);
                cspec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
                cloudtpu::OpNodeSpec mm;
                mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
                mm.src_ids = {1, 2};
                mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
                cspec.ops.push_back(mm);
                cspec.output_ids = {3};

                const auto got = run_emitted(rt, cloudtpu::emit_stablehlo_program(cspec), {bqs, bd, bm, bx});
                CHECK_TRUE(got.size() == static_cast<size_t>(M * N), "Q4_K_I4 cross-block cancellation output size");
                for (float v : got) {
                    CHECK_TRUE(v == 1.0f,
                               "Q4_K_I4 per-block minimum offset must cancel block 0 before reducing across blocks (exact 1.0f)");
                }
            }
        }
    }
    std::printf("  [PASS] Q4_0_I4 & Q4_K_I4 single-read S4/U4/bitcast MUL_MAT (M=1, M=16, M=40 + F32 offset cancellation)\n");
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
            m_s4_f32[static_cast<size_t>(b * N + n)] = m;
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
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0);
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
    const auto copts = rt.make_spmd_compile_options(P, 0, false);

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
                m_s4_km[static_cast<size_t>(b * N + n)] = m;
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

        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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

    // 1b) SHARD_N across P=4 chips with ODD per-chip N_local (N=12 -> N_local=3) in native S4 mode
    {
        const int64_t N = 12, N_local = N / P, NB = 8, K = NB * 32, M = 8;
        std::vector<int8_t> qs_km(static_cast<size_t>(NB * 32 * N));
        std::vector<float> d_km(static_cast<size_t>(NB * N));
        std::vector<float> m_s4_km(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const float d = 0.03f * static_cast<float>((n + b * 2) % 7 + 1);
                const float m = -0.10f + 0.02f * static_cast<float>(n % 5);
                d_km[static_cast<size_t>(b * N + n)] = d;
                m_s4_km[static_cast<size_t>(b * N + n)] = m;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 5 + b * 3 + k * 7) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    qs_km[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.11f);
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
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
        w_spec.sharding = cloudtpu::TensorShardingKind::SHARD_N;
        w_spec.scale_dtype = cloudtpu::PjrtDType::F32;
        spec.inputs.push_back(w_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};
        const auto prog = cloudtpu::emit_stablehlo_program(spec);

        size_t free_before = 0, free_after = 0, total_b = 0;
        CHECK_TRUE(rt.query_memory_stats(0, &free_before, &total_b), "SPMD odd S4 mem before");
        std::vector<std::vector<cloudtpu::PjrtBufferPtr>> per_dev(static_cast<size_t>(P));
        for (int p = 0; p < P; ++p) {
            std::vector<int8_t> qs_s(static_cast<size_t>(NB * 32 * N_local));
            std::vector<float> d_s(static_cast<size_t>(NB * N_local));
            std::vector<float> m_s(static_cast<size_t>(NB * N_local));
            for (int64_t bk = 0; bk < NB * 32; ++bk) {
                for (int64_t nl = 0; nl < N_local; ++nl) {
                    qs_s[static_cast<size_t>(bk * N_local + nl)] =
                        qs_km[static_cast<size_t>(bk * N + p * N_local + nl)];
                }
            }
            for (int64_t b = 0; b < NB; ++b) {
                for (int64_t nl = 0; nl < N_local; ++nl) {
                    d_s[static_cast<size_t>(b * N_local + nl)] = d_km[static_cast<size_t>(b * N + p * N_local + nl)];
                    m_s[static_cast<size_t>(b * N_local + nl)] = m_s4_km[static_cast<size_t>(b * N + p * N_local + nl)];
                }
            }
            per_dev[static_cast<size_t>(p)].push_back(
                rt.buffer_from_host(p, qs_s.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N_local}));
            per_dev[static_cast<size_t>(p)].push_back(
                rt.buffer_from_host(p, d_s.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N_local}));
            per_dev[static_cast<size_t>(p)].push_back(
                rt.buffer_from_host(p, m_s.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N_local}));
            per_dev[static_cast<size_t>(p)].push_back(
                rt.buffer_from_host(p, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K}));
        }
        CHECK_TRUE(rt.query_memory_stats(0, &free_after, &total_b), "SPMD odd S4 mem after");
        const size_t expected_chip0 =
            cloudtpu::pjrt_dtype_nbytes(cloudtpu::PjrtDType::S4, static_cast<size_t>(NB * 32 * N_local)) +
            static_cast<size_t>(NB * N_local) * 2 * sizeof(float) +
            x.size() * sizeof(float);
        CHECK_TRUE((free_before - free_after) == expected_chip0,
                   "SPMD odd-width SHARD_N Q4_K_I4 chip0 uses 0.5B/weight / P");

        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "SPMD odd-width SHARD_N Q4_K_I4 P=4 NMSE < 5e-4");
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
                m_s4_km[static_cast<size_t>(b * N + n)] = m;
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
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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
                const uint16_t mb = cloudtpu::f32_to_bf16(m_raw);
                const float m_bf = cloudtpu::bf16_to_f32(mb);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                m_s4_bf16[static_cast<size_t>(b * N + n)] = mb;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 7 + b * 3 + k * 5) & 0x0F);
                    const int8_t s4 = static_cast<int8_t>(static_cast<int>(u4) - 8);
                    qs_s4[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m_bf;
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
                // M=40 in-loop Q4_K_I4 dequant applies m in F32 before rounding to BF16
                CHECK_TRUE(prog.mlir_text.find("-> tensor<1x1x8x32x16xbf16>") != std::string::npos,
                           "Q4_K_I4 BF16 scales prefill rounds dequantized weight to BF16 after adding m");
            }

            auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
            CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 BF16 scales + pad8 NMSE < 5e-4");
        }

        // 2b) Exact BF16-scale cancellation & pre-rounding offset regressions:
        //   (a) Small-batch factored dot: u4=8 (s4=0), d=1/32, m=-1/4, x=[1, 1/256] -> exact 0.0f
        //   (b) Prefill (M=40) dequant:   u4=3 (s4=-5), d=1.0078125, m=-3.03125, x=[1, 0..] -> exact -0.0078125f
        {
            // (a) Small-batch M=1 with BF16 scales: xsum stays F32 in both scale and offset terms
            const uint16_t d_bf = cloudtpu::f32_to_bf16(0.03125f);   // 1/32 (exact BF16)
            const uint16_t m_bf = cloudtpu::f32_to_bf16(-0.25f);     // -1/4 (exact BF16)
            std::vector<int8_t> qs_0(static_cast<size_t>(NB * 32 * N), 0);  // s4=0 -> u4=8
            std::vector<uint16_t> d_b(static_cast<size_t>(NB * N), d_bf);
            std::vector<uint16_t> m_b(static_cast<size_t>(NB * N), m_bf);
            std::vector<float> x_two(static_cast<size_t>(K), 0.0f);
            x_two[0] = 1.0f;
            x_two[1] = 1.0f / 256.0f;  // exact BF16, but 1 + 1/256 rounds to 1 in BF16!

            auto bqs0 = rt.buffer_from_host(0, qs_0.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
            auto bd0 = rt.buffer_from_host(0, d_b.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
            auto bm0 = rt.buffer_from_host(0, m_b.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
            auto bx0 = rt.buffer_from_host(0, x_two.data(), cloudtpu::PjrtDType::F32, {1, 1, 1, K});

            cloudtpu::GraphProgramSpec sp_a;
            sp_a.bf16_matmul = true;
            sp_a.mm_pad_m = 8;
            cloudtpu::TensorSpec ws_a{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
            ws_a.scale_dtype = cloudtpu::PjrtDType::BF16;
            sp_a.inputs.push_back(ws_a);
            sp_a.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, K}});
            cloudtpu::OpNodeSpec mm_a;
            mm_a.kind = cloudtpu::CloudOpKind::MUL_MAT;
            mm_a.src_ids = {1, 2};
            mm_a.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, N}};
            sp_a.ops.push_back(mm_a);
            sp_a.output_ids = {3};
            const auto got_a = run_emitted(rt, cloudtpu::emit_stablehlo_program(sp_a), {bqs0, bd0, bm0, bx0});
            for (float v : got_a) {
                CHECK_TRUE(v == 0.0f, "Q4_K_I4 BF16-scale small-batch u4=8,d=1/32,m=-1/4 must cancel to exact 0.0f");
            }

            // (b) Prefill M=40 with BF16 scales: u4=3 (s4=-5), d=1.0078125, m=-3.03125 -> -0.0078125f
            const uint16_t dp_bf = cloudtpu::f32_to_bf16(1.0078125f);
            const uint16_t mp_bf = cloudtpu::f32_to_bf16(-3.03125f);
            std::vector<int8_t> qs_m5(static_cast<size_t>(NB * 32 * N), static_cast<int8_t>(-5));  // u4=3
            std::vector<uint16_t> dp_b(static_cast<size_t>(NB * N), dp_bf);
            std::vector<uint16_t> mp_b(static_cast<size_t>(NB * N), mp_bf);
            const int64_t M_pre = 40;
            std::vector<float> x_pre(static_cast<size_t>(M_pre * K), 0.0f);
            x_pre[0] = 1.0f;

            auto bqsp = rt.buffer_from_host(0, qs_m5.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
            auto bdp = rt.buffer_from_host(0, dp_b.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
            auto bmp = rt.buffer_from_host(0, mp_b.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
            auto bxp = rt.buffer_from_host(0, x_pre.data(), cloudtpu::PjrtDType::F32, {1, 1, M_pre, K});

            cloudtpu::GraphProgramSpec sp_b = sp_a;
            sp_b.inputs[1].dims[2] = M_pre;
            sp_b.ops[0].dst.dims[2] = M_pre;
            const auto got_b = run_emitted(rt, cloudtpu::emit_stablehlo_program(sp_b), {bqsp, bdp, bmp, bxp});
            for (int64_t n = 0; n < N; ++n) {
                CHECK_NEAR(got_b[static_cast<size_t>(n)], -0.0078125f, 1e-6f,
                           "Q4_K_I4 BF16-scale prefill must apply m in F32 before rounding to BF16");
            }
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
        const auto copts = srt.make_spmd_compile_options(P, 0, false);

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
            const auto exec = srt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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
            const auto exec = srt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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
    const auto copts = rt.make_spmd_compile_options(1, 0, false);
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

        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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
    const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);

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
            const float m_raw = 0.04f * std::cos(static_cast<float>(n + b * 5) + phase);
            const uint16_t mb = cloudtpu::f32_to_bf16(m_raw);
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
                                                bool dual_out_res_norm = false,
                                                bool use_mixed_embd_set_rows = false) {
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
    uint64_t id_mixed_tbl = 0;
    uint64_t id_mixed_slots = 0;
    if (use_mixed_embd_set_rows) {
        id_mixed_tbl = next_id++;
        id_mixed_slots = next_id++;
        spec.inputs.push_back({id_mixed_tbl, F32, DENSE, {1, 1, T, m.E}, REP});
        spec.inputs.push_back({id_mixed_slots, S32, DENSE, {1, 1, 1, T}, REP});
    }

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
    if (use_mixed_embd_set_rows) {
        cur_x = add_op(cloudtpu::CloudOpKind::SET_ROWS, {cur_x, id_mixed_slots, id_mixed_tbl},
                       F32, 1, 1, T, m.E).dst.id;
    }
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
                                 bool use_mixed_embd,
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
        if (use_mixed_embd) {
            std::vector<float> mixed_tbl(static_cast<size_t>(T * m.E), 0.0f);
            std::vector<int32_t> mixed_slots(static_cast<size_t>(T));
            for (int64_t t = 0; t < T; ++t) {
                mixed_slots[static_cast<size_t>(t)] = static_cast<int32_t>(T - 1 - t);
                for (int64_t e = 0; e < m.E; ++e) {
                    mixed_tbl[static_cast<size_t>(t * m.E + e)] =
                        0.05f * std::cos(static_cast<float>(t * m.E + e) * 0.11f);
                }
            }
            const int64_t d_tbl[4] = {1, 1, T, m.E};
            push_padded(mixed_tbl.data(), cloudtpu::PjrtDType::F32, d_tbl, 6);
            push_padded(mixed_slots.data(), cloudtpu::PjrtDType::S32, d_1d, 7);
        }
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
                        std::vector<int64_t>& active_slots_out,
                        bool use_mixed_embd = false) {
        std::vector<cloudtpu::PjrtBufferPtr> step_in;
        make_step_buffers(T, n_out, mask_rows, bucket_B, plan, use_s64, use_mixed_embd, active_slots_out, step_in);
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
        const auto copts = rt.make_spmd_compile_options(1, 0, false);
        const auto exec = rt.compile_or_get(prog.cache_key, prog.mlir_text, 0, copts);
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

    // 3d) build_inp_embd non-KV SET_ROWS(upd=[1,1,T,E], idx=[1,1,1,T], tbl=[1,1,T,E]) alongside
    //     FLASH_ATTN_EXT KV SET_ROWS([1,1,KV_CAP,D_KV]) across T=5,6,7,8 -> B=8
    {
        auto kv_seq = init_kv_pair();
        const cloudtpu::CacheStats st_before = rt.cache_stats();
        std::string key_ref;
        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7), int64_t(8)}) {
            const int64_t B = cloudtpu::select_batch_bucket(T, {1, 2, 4, 8, 16});
            cloudtpu::GraphProgramSpec spec = build_cb_decode_spec(m, T, T, true, 1, true, true);
            cloudtpu::BucketAnalysisResult plan;
            if (B > T) {
                plan = cloudtpu::analyze_program_token_axes(spec, T, B);
                CHECK_TRUE(plan.success && plan.input_token_axis[6] == 2 &&
                               plan.input_pad_kind[6] == cloudtpu::TokenInputPadKind::ZERO &&
                               plan.input_token_axis[7] == 3 &&
                               plan.input_pad_kind[7] == cloudtpu::TokenInputPadKind::NEG_ONE &&
                               plan.op_dst_token_axis[1] == 2,
                           "build_inp_embd non-KV SET_ROWS must bucket tbl/dst to ax=2 and slots to ax=3 (-1)");
                CHECK_TRUE(cloudtpu::apply_program_batch_bucket(spec, plan), "apply mixed_embd bucket");
                CHECK_TRUE(spec.inputs[6].dims[2] == B && spec.ops[1].dst.dims[2] == B,
                           "mixed_embd tbl and SET_ROWS dst must resize to [1,1,B,E]");
                CHECK_TRUE(spec.inputs[12].dims[2] == m.KV_CAP,
                           "KV cache input must stay at [1,1,KV_CAP,D_KV]");
            }
            const cloudtpu::EmittedProgram prog = cloudtpu::emit_stablehlo_program(spec);
            if (key_ref.empty()) {
                key_ref = prog.cache_key;
            } else {
                CHECK_TRUE(prog.cache_key == key_ref,
                           "mixed_embd SET_ROWS T=5,6,7,8 bucketed specs must share identical cache_key");
            }
            std::vector<int64_t> active_slots;
            run_step(spec, prog, T, 1, T, B, (B > T) ? &plan : nullptr, true, kv_seq, active_slots, true);
        }
        const cloudtpu::CacheStats st_after = rt.cache_stats();
        CHECK_TRUE(st_after.misses - st_before.misses == 1,
                   "mixed_embd SET_ROWS T=5,6,7,8 must have misses == 1");
        CHECK_TRUE(st_after.hits - st_before.hits == 3,
                   "mixed_embd SET_ROWS T=5,6,7,8 must have hits == 3");

        for (int64_t T : {int64_t(5), int64_t(6), int64_t(7)}) {
            auto kv_buck = init_kv_pair();
            auto kv_unb = init_kv_pair();
            cloudtpu::GraphProgramSpec spec_u = build_cb_decode_spec(m, T, T, true, 1, true, true);
            const cloudtpu::EmittedProgram prog_u = cloudtpu::emit_stablehlo_program(spec_u);
            std::vector<int64_t> slots_u;
            const std::vector<float> logits_u =
                run_step(spec_u, prog_u, T, 1, T, T, nullptr, true, kv_unb, slots_u, true);

            cloudtpu::GraphProgramSpec spec_b = spec_u;
            const cloudtpu::BucketAnalysisResult plan = cloudtpu::analyze_program_token_axes(spec_b, T, 8);
            CHECK_TRUE(plan.success && cloudtpu::apply_program_batch_bucket(spec_b, plan),
                       "mixed_embd SET_ROWS bucket parity");
            const cloudtpu::EmittedProgram prog_b = cloudtpu::emit_stablehlo_program(spec_b);
            std::vector<int64_t> slots_b;
            const std::vector<float> logits_b =
                run_step(spec_b, prog_b, T, 1, T, 8, &plan, true, kv_buck, slots_b, true);
            for (size_t i = 0; i < logits_b.size(); ++i) {
                CHECK_NEAR(logits_b[i], logits_u[i], 1e-4f,
                           "mixed_embd SET_ROWS bucketed logits parity < 1e-4 vs unbucketed");
            }
            verify_kv_preservation(kv_buck, kv_unb, slots_b);
        }

        // Verify tiny donated KV cache with KV_CAP == T is still protected by is_kv while mixed_embd buckets
        CbModelDims m_tiny = m;
        m_tiny.KV_CAP = 5;
        m_tiny.N_KV = 4;
        const cloudtpu::GraphProgramSpec tiny_spec = build_cb_decode_spec(m_tiny, 5, 5, true, 1, true, true);
        const cloudtpu::BucketAnalysisResult tiny_plan = cloudtpu::analyze_program_token_axes(tiny_spec, 5, 8);
        CHECK_TRUE(tiny_plan.success && tiny_plan.input_token_axis[6] == 2 &&
                       tiny_plan.input_token_axis[12] == -1 && tiny_plan.input_token_axis[13] == -1,
                   "tiny KV_CAP==T cache must stay unbucketed while mixed_embd buckets to ax=2");
    }

    // 4) Reshaped S64 SET_ROWS index propagates NEG_ONE to the underlying input (0 overwrite of KV row 0),
    //    while non-shape computed indices and ZERO/NEG_ONE pad conflicts fail closed.
    {
        const int64_t T = 5, B = 8, KV_R = 16, D_R = 4;
        cloudtpu::GraphProgramSpec rspec;
        rspec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, D_R}});
        rspec.inputs.push_back({2, cloudtpu::PjrtDType::S64, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, 1}});
        cloudtpu::TensorSpec kv_in{3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        kv_in.alias_output_idx = 0;
        rspec.inputs.push_back(kv_in);

        cloudtpu::OpNodeSpec rsh;
        rsh.kind = cloudtpu::CloudOpKind::RESHAPE;
        rsh.src_ids = {2};
        rsh.dst = {4, cloudtpu::PjrtDType::S64, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, T}};
        rspec.ops.push_back(rsh);

        cloudtpu::OpNodeSpec sr;
        sr.kind = cloudtpu::CloudOpKind::SET_ROWS;
        sr.src_ids = {1, 4, 3};
        sr.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        rspec.ops.push_back(sr);
        rspec.output_ids = {5};

        const cloudtpu::BucketAnalysisResult rplan = cloudtpu::analyze_program_token_axes(rspec, T, B);
        CHECK_TRUE(rplan.success, "reshaped S64 SET_ROWS index must bucket cleanly");
        CHECK_TRUE(rplan.input_token_axis[1] == 2 && rplan.input_pad_kind[1] == cloudtpu::TokenInputPadKind::NEG_ONE,
                   "reshaped S64 SET_ROWS index must propagate NEG_ONE pad kind to underlying input");

        cloudtpu::GraphProgramSpec bspec = rspec;
        CHECK_TRUE(cloudtpu::apply_program_batch_bucket(bspec, rplan), "apply_program_batch_bucket for reshaped SET_ROWS");
        const cloudtpu::EmittedProgram bprog = cloudtpu::emit_stablehlo_program(bspec);

        std::vector<float> upd(static_cast<size_t>(T * D_R));
        for (size_t i = 0; i < upd.size(); ++i) upd[i] = 10.0f + static_cast<float>(i);
        const std::vector<int64_t> raw_idx = {2, 5, 7, 10, 12};
        std::vector<float> kv_init(static_cast<size_t>(KV_R * D_R));
        for (size_t i = 0; i < kv_init.size(); ++i) kv_init[i] = -77.5f - static_cast<float>(i);

        const int64_t upd_orig[4] = {1, 1, T, D_R};
        const int64_t upd_pdims[4] = {1, 1, B, D_R};
        std::vector<float> upd_pad(static_cast<size_t>(B * D_R), 0.0f);
        cloudtpu::pad_dense_host_tensor(
            upd.data(), upd_pad.data(), cloudtpu::PjrtDType::F32, upd_orig, upd_pdims,
            rplan.input_token_axis[0], rplan.input_pad_kind[0]);

        const int64_t idx_orig[4] = {1, 1, T, 1};
        const int64_t idx_pdims[4] = {1, 1, B, 1};
        std::vector<int64_t> idx_pad(static_cast<size_t>(B), 0);
        cloudtpu::pad_dense_host_tensor(
            raw_idx.data(), idx_pad.data(), cloudtpu::PjrtDType::S64, idx_orig, idx_pdims,
            rplan.input_token_axis[1], rplan.input_pad_kind[1]);
        auto b_upd = rt.buffer_from_host(0, upd_pad.data(), cloudtpu::PjrtDType::F32, {1, 1, B, D_R});
        auto b_idx = rt.buffer_from_host(0, idx_pad.data(), cloudtpu::PjrtDType::S64, {1, 1, B, 1});
        auto b_kv = rt.buffer_from_host(0, kv_init.data(), cloudtpu::PjrtDType::F32, {1, 1, KV_R, D_R});

        const auto copts = rt.make_spmd_compile_options(1, 0, false);
        const auto exec = rt.compile_or_get(bprog.cache_key, bprog.mlir_text, 0, copts);
        std::vector<PJRT_Buffer*> raw_in = {b_upd->handle.get(), b_idx->handle.get(), b_kv->handle.get()};
        std::vector<cloudtpu::PjrtBufferPtr> outs;
        rt.execute(exec, 0, raw_in.data(), raw_in.size(), outs,
                   {cloudtpu::PjrtDType::F32}, {{1, 1, KV_R, D_R}}, true);
        std::vector<float> kv_after(static_cast<size_t>(KV_R * D_R), 0.0f);
        rt.buffer_to_host(outs[0]->handle.get(), kv_after.data(), kv_after.size() * sizeof(float));

        for (int64_t r = 0; r < KV_R; ++r) {
            int64_t match_t = -1;
            for (int64_t t = 0; t < T; ++t) {
                if (raw_idx[static_cast<size_t>(t)] == r) match_t = t;
            }
            for (int64_t d = 0; d < D_R; ++d) {
                const size_t off = static_cast<size_t>(r * D_R + d);
                const float exp = (match_t >= 0) ? upd[static_cast<size_t>(match_t * D_R + d)] : kv_init[off];
                CHECK_NEAR(kv_after[off], exp, 1e-5f,
                           "reshaped SET_ROWS index NEG_ONE padding must preserve KV row 0 and all untouched rows");
            }
        }

        // Non-shape computed index (ADD) must fail closed instead of zero-padding and overwriting KV row 0
        cloudtpu::GraphProgramSpec cspec;
        cspec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, D_R}});
        cspec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, T}});
        cspec.inputs.push_back({3, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 1}});
        cloudtpu::TensorSpec ckv{4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        ckv.alias_output_idx = 0;
        cspec.inputs.push_back(ckv);
        cloudtpu::OpNodeSpec add_idx;
        add_idx.kind = cloudtpu::CloudOpKind::ADD;
        add_idx.src_ids = {2, 3};
        add_idx.dst = {5, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, T}};
        cspec.ops.push_back(add_idx);
        cloudtpu::OpNodeSpec csr;
        csr.kind = cloudtpu::CloudOpKind::SET_ROWS;
        csr.src_ids = {1, 5, 4};
        csr.dst = {6, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        cspec.ops.push_back(csr);
        cspec.output_ids = {6};
        const cloudtpu::BucketAnalysisResult cplan = cloudtpu::analyze_program_token_axes(cspec, T, B);
        CHECK_TRUE(!cplan.success && cplan.fail_reason.find("neg_one_computed_index") != std::string::npos,
                   "computed ADD SET_ROWS index must fail closed with neg_one_computed_index");

        // Conflicting ZERO (ROPE pos) + NEG_ONE (SET_ROWS idx) on the same reshaped input must fail closed
        cloudtpu::GraphProgramSpec conflict_spec;
        conflict_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, T, 1, D_R}});
        conflict_spec.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, 1}});
        cloudtpu::TensorSpec xkv{3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        xkv.alias_output_idx = 0;
        conflict_spec.inputs.push_back(xkv);
        cloudtpu::OpNodeSpec r_pos;
        r_pos.kind = cloudtpu::CloudOpKind::RESHAPE;
        r_pos.src_ids = {2};
        r_pos.dst = {4, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, T}};
        conflict_spec.ops.push_back(r_pos);
        cloudtpu::OpNodeSpec rope_op;
        rope_op.kind = cloudtpu::CloudOpKind::ROPE;
        rope_op.src_ids = {1, 4};
        rope_op.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, T, 1, D_R}};
        rope_op.rope.n_dims = static_cast<int32_t>(D_R);
        conflict_spec.ops.push_back(rope_op);
        cloudtpu::OpNodeSpec flat_op;
        flat_op.kind = cloudtpu::CloudOpKind::RESHAPE;
        flat_op.src_ids = {5};
        flat_op.dst = {6, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, D_R}};
        conflict_spec.ops.push_back(flat_op);
        cloudtpu::OpNodeSpec xsr;
        xsr.kind = cloudtpu::CloudOpKind::SET_ROWS;
        xsr.src_ids = {6, 4, 3};
        xsr.dst = {7, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, KV_R, D_R}};
        conflict_spec.ops.push_back(xsr);
        conflict_spec.output_ids = {7};
        const cloudtpu::BucketAnalysisResult xplan = cloudtpu::analyze_program_token_axes(conflict_spec, T, B);
        CHECK_TRUE(!xplan.success && xplan.fail_reason.find("pad_kind_conflict") != std::string::npos,
                   "shared reshaped input with ZERO (ROPE) and NEG_ONE (SET_ROWS) must fail closed");
    }

    // 5) SOFT_MAX with active ALiBi (max_bias > 0) must reject head-axis (ax==1) bucketing so ALiBi slopes
    //    stay invariant to padding, while non-ALiBi and non-head-axis SOFT_MAX still bucket cleanly.
    {
        const int64_t T = 3, B = 4, D = 8;
        auto make_alibi_head_spec = [&](float max_bias) {
            cloudtpu::GraphProgramSpec sp;
            sp.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, D}});
            sp.inputs.push_back({2, cloudtpu::PjrtDType::S32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, T}});
            sp.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, D}});
            cloudtpu::OpNodeSpec gr;
            gr.kind = cloudtpu::CloudOpKind::GET_ROWS;
            gr.src_ids = {1, 2};
            gr.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, D}};
            sp.ops.push_back(gr);
            cloudtpu::OpNodeSpec rsh;
            rsh.kind = cloudtpu::CloudOpKind::RESHAPE;
            rsh.src_ids = {4};
            rsh.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, T, 1, D}};
            sp.ops.push_back(rsh);
            cloudtpu::OpNodeSpec sm;
            sm.kind = cloudtpu::CloudOpKind::SOFT_MAX;
            sm.src_ids = {5, 3};
            sm.soft_max.scale = 1.0f;
            sm.soft_max.max_bias = max_bias;
            sm.dst = {6, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, T, 1, D}};
            sp.ops.push_back(sm);
            sp.output_ids = {6};
            return sp;
        };

        const auto alibi_head = cloudtpu::analyze_program_token_axes(make_alibi_head_spec(8.0f), T, B);
        CHECK_TRUE(!alibi_head.success && alibi_head.fail_reason.find("softmax_alibi_head_axis") != std::string::npos,
                   "ALiBi SOFT_MAX with token axis on head dim (ax==1) must reject bucketing");

        const auto non_alibi_head = cloudtpu::analyze_program_token_axes(make_alibi_head_spec(0.0f), T, B);
        CHECK_TRUE(non_alibi_head.success && non_alibi_head.output_token_axis[0] == 1,
                   "non-ALiBi SOFT_MAX on ax==1 must still bucket cleanly");

        // ALiBi SOFT_MAX with token axis on query dim (ax==2, fixed 4 heads on ax==1) must still bucket cleanly
        cloudtpu::GraphProgramSpec alibi_q;
        alibi_q.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 4, T, D}});
        alibi_q.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, T, D}});
        alibi_q.inputs.push_back({3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, D, D}});
        cloudtpu::OpNodeSpec sm_q;
        sm_q.kind = cloudtpu::CloudOpKind::SOFT_MAX;
        sm_q.src_ids = {1, 2};
        sm_q.soft_max.scale = 1.0f;
        sm_q.soft_max.max_bias = 8.0f;
        sm_q.dst = {4, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 4, T, D}};
        alibi_q.ops.push_back(sm_q);
        cloudtpu::OpNodeSpec mm_q;
        mm_q.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm_q.src_ids = {3, 4};
        mm_q.dst = {5, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 4, T, D}};
        alibi_q.ops.push_back(mm_q);
        alibi_q.output_ids = {5};
        const auto alibi_q_plan = cloudtpu::analyze_program_token_axes(alibi_q, T, B);
        CHECK_TRUE(alibi_q_plan.success && alibi_q_plan.output_token_axis[0] == 2,
                   "ALiBi SOFT_MAX with token axis on ax==2 (fixed heads) must bucket cleanly");
    }

    // 6) Oversized CLOUDTPU_BATCH_BUCKETS rejection & overflow-checked padded tensor byte sizing
    {
        const std::vector<int64_t> parsed = cloudtpu::parse_batch_buckets(
            "1,8,131072,131073,99999999999999999999999,-4,0,abc");
        const std::vector<int64_t> expected = {1, 8, cloudtpu::kMaxBatchBucketTokens};
        CHECK_TRUE(parsed == expected,
                   "parse_batch_buckets must reject entries > kMaxBatchBucketTokens, ERANGE, <=0, and invalid tokens");

        size_t nb = 0;
        const int64_t ok_dims[4] = {1, 4, 8, 64};
        CHECK_TRUE(cloudtpu::checked_tensor_nbytes(ok_dims, 4, &nb) && nb == 8192U,
                   "checked_tensor_nbytes valid shape");
        const int64_t neg_dims[4] = {1, -2, 8, 64};
        CHECK_TRUE(!cloudtpu::checked_tensor_nbytes(neg_dims, 4, &nb),
                   "checked_tensor_nbytes rejects negative dims");
        CHECK_TRUE(!cloudtpu::checked_tensor_nbytes(ok_dims, 0, &nb),
                   "checked_tensor_nbytes rejects zero elem_bytes");
        const int64_t ovf_dims[4] = {1LL << 30, 1LL << 30, 16, 1};
        CHECK_TRUE(!cloudtpu::checked_tensor_nbytes(ovf_dims, 4, &nb),
                   "checked_tensor_nbytes rejects size_t multiplication overflow");

        cloudtpu::GraphProgramSpec ov_spec;
        ov_spec.inputs.push_back({1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 16}});
        ov_spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, 16}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {2, 1};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 5, 16}};
        ov_spec.ops.push_back(mm);
        ov_spec.output_ids = {3};
        const auto too_big = cloudtpu::analyze_program_token_axes(ov_spec, 5, cloudtpu::kMaxBatchBucketTokens + 1);
        CHECK_TRUE(!too_big.success && too_big.fail_reason == "bucket_tokens_too_large",
                   "analyze_program_token_axes must reject bucket_tokens > kMaxBatchBucketTokens");

        cloudtpu::GraphProgramSpec ov_mm_spec;
        ov_mm_spec.inputs.push_back(
            {1, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1LL << 26, 1LL << 26, 5, 16}});
        ov_mm_spec.inputs.push_back(
            {2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 16, 16}});
        cloudtpu::OpNodeSpec ov_mm;
        ov_mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        ov_mm.src_ids = {2, 1};
        ov_mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1LL << 26, 1LL << 26, 5, 16}};
        ov_mm_spec.ops.push_back(ov_mm);
        ov_mm_spec.output_ids = {3};
        const auto ov_plan = cloudtpu::analyze_program_token_axes(ov_mm_spec, 5, 8192);
        CHECK_TRUE(!ov_plan.success && ov_plan.fail_reason == "padded_size_overflow",
                   "analyze_program_token_axes must fail closed on padded tensor byte overflow");
    }

    rt.shutdown();
    std::printf("  [PASS] Continuous batching padded buckets (T=5..8 -> B=8 + N_out<T + mixed_embd + reshaped SET_ROWS + ALiBi + overflow guards: misses=1, hits=3, parity < 1e-4, 0 KV corruption)\n");
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
    ScopedEnv env_bf16("CLOUDTPU_MATMUL_BF16", "0");            // Isolate fusion/donation parity (< 1e-4)

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

    // 5) Default CLOUDTPU_MATMUL_BF16 (unset) still fuses into subgraphs == 1 & passes bounded parity
    {
        ScopedEnv env_bf16_default("CLOUDTPU_MATMUL_BF16", nullptr);
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
    auto fill_q6k = [](size_t /*nrows*/, size_t nbytes) {
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
        auto step_kv_slot = [](size_t step_idx, int64_t t) -> int64_t {
            static const int64_t base_slots[8] = {2, 5, 8, 11, 14, 17, 20, 22};
            // Shift existing lanes on cache-hit steps (step_idx > 0) across distinct slots in
            // [1, N_KV-1] while keeping slot 0 and slots [N_KV, KV_CAP) untouched.
            return (step_idx == 0)
                       ? base_slots[static_cast<size_t>(t)]
                       : (1 + ((base_slots[static_cast<size_t>(t)] + static_cast<int64_t>(step_idx)) % 23));
        };
        for (size_t step_idx = 0; step_idx < batch_sizes.size(); ++step_idx) {
            const int64_t T = batch_sizes[step_idx];
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
                const int64_t kv_slot = step_kv_slot(step_idx, t);
                tokens[static_cast<size_t>(t)] =
                    static_cast<int32_t>((t * 5 + static_cast<int64_t>(step_idx) * 7 + 3) % V_emb);
                pos[static_cast<size_t>(t)] =
                    static_cast<int32_t>((t + 1) + static_cast<int64_t>(step_idx) * 4);
                row_idx64[static_cast<size_t>(t)] = kv_slot;
                out_ids[static_cast<size_t>(t)] =
                    static_cast<int32_t>((T - 1 - t + static_cast<int64_t>(step_idx)) % T);
                mask_f16[static_cast<size_t>(t * N_KV + 0)] = cloudtpu::f32_to_f16(0.0f);
                mask_f16[static_cast<size_t>(t * N_KV + kv_slot)] = cloudtpu::f32_to_f16(0.0f);
                if (step_idx > 0) {
                    const int64_t prev_slot = step_kv_slot(step_idx - 1, t);
                    mask_f16[static_cast<size_t>(t * N_KV + prev_slot)] = cloudtpu::f32_to_f16(0.0f);
                }
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
                    const float p_val = y_row1[static_cast<size_t>(v)];
                    const float f_val = y[static_cast<size_t>(V_out + v)];
                    if (!std::isfinite(p_val) || !std::isfinite(f_val) ||
                        std::fabs(p_val - f_val) > 1e-6f) {
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

    // 1) T=5,6,7,8 sequence with default CLOUDTPU_BATCH_BUCKETS -> plan_misses=1, plan_hits=3.
    //    Inputs (token, pos, attention mask, KV destination, out_ids) vary for existing lanes on
    //    every cache-hit step; verify every step's logits and the final KV state match the
    //    identical T=5,6,7,8 sequence run with bucketing disabled.
    const std::vector<int64_t> seq_batches = {5, 6, 7, 8};
    const int64_t fixed_slots[8] = {2, 5, 8, 11, 14, 17, 20, 22};
    auto seq_kv_slot = [&](size_t step_idx, int64_t t) -> int64_t {
        return (step_idx == 0)
                   ? fixed_slots[static_cast<size_t>(t)]
                   : (1 + ((fixed_slots[static_cast<size_t>(t)] + static_cast<int64_t>(step_idx)) % 23));
    };
    const CbRunResult seq_res = run_steps(seq_batches, nullptr);
    const CbRunResult seq_unbucketed = run_steps(seq_batches, "0");
    CHECK_TRUE(seq_res.ok && seq_unbucketed.ok, "ggml CB bucket sequence T=5,6,7,8 succeeded");
    const uint64_t misses = parse_log_u64(seq_res.logs, "plan_misses");
    const uint64_t hits = parse_log_u64(seq_res.logs, "plan_hits");
    const uint64_t bucketed = parse_log_u64(seq_res.logs, "bucketed");
    CHECK_TRUE(misses == 1 && hits == 3, "ggml CB T=5,6,7,8 must yield plan_misses==1, plan_hits==3");
    CHECK_TRUE(bucketed == 3, "ggml CB T=5,6,7 must increment bucketed==3");
    CHECK_TRUE(parse_log_u64(seq_unbucketed.logs, "bucketed") == 0,
               "ggml CB unbucketed reference T=5,6,7,8 must have bucketed==0");
    CHECK_TRUE(seq_res.logits_per_step.size() == 4 && seq_unbucketed.logits_per_step.size() == 4,
               "ggml CB T=5,6,7,8 must return 4 step logit buffers");
    float max_step01_lane_delta = 0.0f;
    for (int64_t i = 0; i < 5 * V_out; ++i) {
        max_step01_lane_delta = std::max(
            max_step01_lane_delta,
            std::fabs(seq_res.logits_per_step[1][static_cast<size_t>(i)] -
                      seq_res.logits_per_step[0][static_cast<size_t>(i)]));
    }
    CHECK_TRUE(max_step01_lane_delta > 1e-3f,
               "ggml CB cache-hit step must produce updated logits for varied existing lanes");
    for (size_t s = 0; s < seq_res.logits_per_step.size(); ++s) {
        CHECK_TRUE(seq_res.logits_per_step[s].size() == seq_unbucketed.logits_per_step[s].size(),
                   "ggml CB sequence step logit size must match unbucketed");
        for (size_t i = 0; i < seq_res.logits_per_step[s].size(); ++i) {
            CHECK_TRUE(std::isfinite(seq_res.logits_per_step[s][i]) &&
                           std::isfinite(seq_unbucketed.logits_per_step[s][i]),
                       "ggml CB sequence logits must be finite");
            CHECK_NEAR(seq_res.logits_per_step[s][i], seq_unbucketed.logits_per_step[s][i], 1e-4f,
                       "ggml CB sequence logits (incl. cache hits) must match unbucketed < 1e-4");
        }
    }
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t slot = 0; slot < KV_CAP; ++slot) {
            bool active = false;
            for (size_t s = 0; s < seq_batches.size() && !active; ++s) {
                for (int64_t t = 0; t < seq_batches[s]; ++t) {
                    if (seq_kv_slot(s, t) == slot) {
                        active = true;
                        break;
                    }
                }
            }
            for (int64_t d = 0; d < D_KV; ++d) {
                const size_t off = static_cast<size_t>(slot * D_KV + d);
                if (!active) {
                    CHECK_TRUE(seq_res.kc_after[static_cast<size_t>(l)][off] == sentinel_kv[off] &&
                                   seq_res.vc_after[static_cast<size_t>(l)][off] == sentinel_kv[off],
                               "ggml CB sequence unused KV slots must remain untouched");
                }
                CHECK_TRUE(seq_res.kc_after[static_cast<size_t>(l)][off] ==
                               seq_unbucketed.kc_after[static_cast<size_t>(l)][off] &&
                               seq_res.vc_after[static_cast<size_t>(l)][off] ==
                                   seq_unbucketed.vc_after[static_cast<size_t>(l)][off],
                           "ggml CB sequence final KV state after cache hits must match unbucketed");
            }
        }
    }

    // 2) Single-step parity (< 1e-4) vs unbucketed (CLOUDTPU_BATCH_BUCKETS=0) and zero KV corruption
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

void test_ggml_cgraph_odd_s4_shard_n() {
#if !defined(_WIN32)
    // Run each CLOUDTPU_Q4_MODE in a fresh process because ggml_backend_cloudtpu_reg()
    // latches process-wide SPMD/Q4 environment settings on first initialization.
    const int64_t K = 256, N = 12, M = 4;
    const size_t w_nbytes = static_cast<size_t>(N) * 144U;
    std::vector<uint8_t> w_raw(w_nbytes, 0);
    for (size_t b = 0; b < static_cast<size_t>(N); ++b) {
        uint8_t* p = w_raw.data() + b * 144U;
        const uint16_t d = cloudtpu::f32_to_f16(0.02f * static_cast<float>((b % 5) + 1));
        const uint16_t dm = cloudtpu::f32_to_f16(0.01f * static_cast<float>((b % 3) + 1));
        std::memcpy(p + 0, &d, 2);
        std::memcpy(p + 2, &dm, 2);
        for (size_t i = 0; i < 12; ++i) p[4 + i] = static_cast<uint8_t>((b * 3 + i * 5 + 1) & 0x3F);
        for (size_t i = 0; i < 128; ++i) p[16 + i] = static_cast<uint8_t>((b * 11 + i * 7) & 0xFF);
    }
    std::vector<float> x_host(static_cast<size_t>(M * K));
    for (size_t i = 0; i < x_host.size(); ++i) x_host[i] = std::sin(static_cast<float>(i + 1) * 0.13f);

    auto run_mode = [&](ggml_backend_t be, size_t* w_hbm_delta_out) -> std::vector<float> {
        std::vector<float> out_y;
        const size_t mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 4096;
        struct ggml_init_params wp = {mem, nullptr, true};
        struct ggml_context* ctx_w = ggml_init(wp);
        ggml_tensor* tw = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, K, N);
        ggml_set_name(tw, "output.weight");
        size_t free_before = 0, free_after = 0, total_b = 0;
        if (w_hbm_delta_out) {
            ggml_backend_dev_memory(ggml_backend_get_device(be), &free_before, &total_b);
        }
        ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, be);
        ggml_backend_tensor_set(tw, w_raw.data(), 0, w_raw.size());
        if (w_hbm_delta_out) {
            ggml_backend_dev_memory(ggml_backend_get_device(be), &free_after, &total_b);
            *w_hbm_delta_out = (free_before >= free_after) ? (free_before - free_after) : 0;
        }

        struct ggml_context* ctx_g = ggml_init(wp);
        struct ggml_cgraph* gf = ggml_new_graph(ctx_g);
        ggml_tensor* tx = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, K, M);
        ggml_set_input(tx);
        ggml_tensor* ty = ggml_mul_mat(ctx_g, tw, tx);
        ggml_set_output(ty);
        ggml_build_forward_expand(gf, ty);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (ggml_gallocr_alloc_graph(galloc, gf)) {
            ggml_backend_tensor_set(tx, x_host.data(), 0, x_host.size() * sizeof(float));
            if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
                out_y.resize(static_cast<size_t>(M * N), 0.0f);
                ggml_backend_tensor_get(ty, out_y.data(), 0, out_y.size() * sizeof(float));
            }
        }
        ggml_gallocr_free(galloc);
        ggml_free(ctx_g);
        ggml_backend_buffer_free(w_buf);
        ggml_free(ctx_w);
        return out_y;
    };

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    CHECK_TRUE(cpu_be != nullptr, "cpu init for odd S4 SHARD_N test");
    const std::vector<float> ref = run_mode(cpu_be, nullptr);
    ggml_backend_free(cpu_be);
    CHECK_TRUE(ref.size() == static_cast<size_t>(M * N), "cpu ref for odd S4 SHARD_N");

    struct ModeCase {
        const char* mode;
        size_t expected_chip0_bytes;
    };
    // K=256 (< 4*256 so SHARD_K is unavailable), N=12 (N/4 = 3 odd):
    // native i4/u4 SHARD_N uses 3 rows/chip (3 * 192B = 576B);
    // bitcast keeps even-width restriction and replicates all 12 rows (12 * 192B = 2304B).
    const ModeCase cases[] = {
        {"i4", 576U},
        {"u4", 576U},
        {"bitcast", 2304U},
    };
    std::fflush(stdout);
    std::fflush(stderr);
    for (const ModeCase& tc : cases) {
        const pid_t pid = fork();
        CHECK_TRUE(pid >= 0, "fork for isolated ggml_backend_cloudtpu_reg test");
        if (pid == 0) {
            setenv("CLOUDTPU_MOCK", "1", 1);
            setenv("CLOUDTPU_MOCK_DEVICES", "4", 1);
            setenv("CLOUDTPU_SPMD_CHIPS", "4", 1);
            setenv("CLOUDTPU_SPMD_MIN_ROWS", "8", 1);
            setenv("CLOUDTPU_Q4K_BF16_SCALES", "0", 1);
            setenv("CLOUDTPU_Q4_MODE", tc.mode, 1);
            ggml_backend_t be = ggml_backend_cloudtpu_init();
            size_t delta = 0;
            bool ok = false;
            if (be != nullptr) {
                const std::vector<float> got = run_mode(be, &delta);
                ggml_backend_free(be);
                ok = (got.size() == ref.size() && compute_nmse(got, ref) < 5e-4 &&
                      delta == tc.expected_chip0_bytes);
            }
            _exit(ok ? 0 : 1);
        }
        int status = 0;
        CHECK_TRUE(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                   "odd-width Q4_K SHARD_N child check failed (i4/u4 must shard 1/4 HBM; bitcast replicates)");
    }
    std::printf("  [PASS] ggml_cgraph odd-width SHARD_N (N=12/P=4 -> 3 rows/chip: i4/u4=576B sharded, bitcast=2304B replicated)\n");
#endif
}

void test_ggml_cgraph_reshaped_host_cpy_and_q4_bitcast_view() {
#if !defined(_WIN32)
    // 1) Bucketed CPY into a reshaped host view [1,1,5,32] over host root [1,5,4,8]
    //    must slice using canonical root shape [1,5,4,8] (axis 1: 8 -> 5) without shape mismatch.
    {
        const int64_t T = 5, D = 32;
        std::vector<float> w_data(static_cast<size_t>(D * D));
        std::vector<float> x_data(static_cast<size_t>(T * D));
        for (size_t i = 0; i < w_data.size(); ++i) w_data[i] = 0.05f * std::sin(static_cast<float>(i + 1) * 0.19f);
        for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = 0.25f * std::cos(static_cast<float>(i + 1) * 0.13f);
        std::vector<float> ref(static_cast<size_t>(T * D), 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t o = 0; o < D; ++o) {
                float acc = 0.0f;
                for (int64_t k = 0; k < D; ++k) {
                    acc += x_data[static_cast<size_t>(t * D + k)] * w_data[static_cast<size_t>(o * D + k)];
                }
                ref[static_cast<size_t>(t * D + o)] = acc;
            }
        }
        std::fflush(stdout);
        std::fflush(stderr);
        const pid_t pid = fork();
        CHECK_TRUE(pid >= 0, "fork for reshaped host CPY bucket test");
        if (pid == 0) {
            setenv("CLOUDTPU_MOCK", "1", 1);
            setenv("CLOUDTPU_MOCK_DEVICES", "1", 1);
            setenv("CLOUDTPU_SPMD_CHIPS", "1", 1);
            setenv("CLOUDTPU_BATCH_BUCKETS", "1,2,4,8", 1);
            setenv("CLOUDTPU_VERBOSE", "1", 1);
            GgmlLogCapture cap;
            ggml_backend_t be = ggml_backend_cloudtpu_init();
            bool ok = false;
            if (be != nullptr) {
                const size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096;
                struct ggml_init_params p = {mem, nullptr, true};
                struct ggml_context* ctx_w = ggml_init(p);
                ggml_tensor* tw = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, D, D);
                ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, be);
                ggml_backend_tensor_set(tw, w_data.data(), 0, w_data.size() * sizeof(float));

                std::vector<float> host_dst(static_cast<size_t>(T * D), -999.0f);
                struct ggml_context* ctx_g = ggml_init(p);
                struct ggml_cgraph* gf = ggml_new_graph(ctx_g);
                ggml_tensor* tx = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, D, T);
                ggml_set_input(tx);
                ggml_tensor* ty = ggml_mul_mat(ctx_g, tw, tx);

                ggml_tensor* dst_root = ggml_new_tensor_4d(ctx_g, GGML_TYPE_F32, 8, 4, T, 1);
                dst_root->data = host_dst.data();
                ggml_tensor* dst_view = ggml_reshape_2d(ctx_g, dst_root, D, T);
                ggml_tensor* cpy = ggml_cpy(ctx_g, ty, dst_view);
                ggml_set_output(cpy);
                ggml_build_forward_expand(gf, cpy);

                ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (ggml_gallocr_alloc_graph(galloc, gf)) {
                    ggml_backend_tensor_set(tx, x_data.data(), 0, x_data.size() * sizeof(float));
                    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
                        ok = true;
                        for (size_t i = 0; i < ref.size(); ++i) {
                            if (std::fabs(host_dst[i] - ref[i]) > 1e-4f || !std::isfinite(host_dst[i])) {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                ggml_gallocr_free(galloc);
                ggml_free(ctx_g);
                ggml_backend_buffer_free(w_buf);
                ggml_free(ctx_w);
                ggml_backend_free(be);
                if (parse_log_u64(cap.buf, "bucketed") != 1) {
                    ok = false;
                }
            }
            _exit(ok ? 0 : 1);
        }
        int status = 0;
        CHECK_TRUE(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                   "bucketed CPY into reshaped host view must slice with canonical root shape");
    }

    // 2) 3-row Q4_K tensor aliasing offset 0 of a 4-row BITCAST Q4_K slot on 1 chip must resolve Q4_K_SPLIT
    //    (not inherit incompatible packed U8 BITCAST layout from the 4-row slot at the same offset).
    {
        const int64_t K = 256, N_FULL = 4, N_VIEW = 3, M = 2;
        std::vector<uint8_t> w_raw(static_cast<size_t>(N_FULL) * 144U, 0);
        for (size_t b = 0; b < static_cast<size_t>(N_FULL); ++b) {
            uint8_t* p = w_raw.data() + b * 144U;
            const uint16_t d = cloudtpu::f32_to_f16(0.02f * static_cast<float>(b + 1));
            const uint16_t dm = cloudtpu::f32_to_f16(0.01f * static_cast<float>(b + 1));
            std::memcpy(p + 0, &d, 2);
            std::memcpy(p + 2, &dm, 2);
            for (size_t i = 0; i < 12; ++i) p[4 + i] = static_cast<uint8_t>((b * 5 + i * 3 + 1) & 0x3F);
            for (size_t i = 0; i < 128; ++i) p[16 + i] = static_cast<uint8_t>((b * 13 + i * 7) & 0xFF);
        }
        std::vector<float> x_host(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x_host.size(); ++i) x_host[i] = std::sin(static_cast<float>(i + 1) * 0.11f);

        // Compute real CPU reference matmul using the first 3 rows of w_raw
        std::vector<float> ref_y;
        {
            ggml_backend_t cpu_be = ggml_backend_cpu_init();
            CHECK_TRUE(cpu_be != nullptr, "cpu init for 3-row BITCAST Q4_K alias reference");
            const size_t mem = ggml_tensor_overhead() * 24 + ggml_graph_overhead() + 4096;
            struct ggml_init_params p = {mem, nullptr, true};
            struct ggml_context* ctx_w = ggml_init(p);
            ggml_tensor* tw3_cpu = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, K, N_VIEW);
            ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, cpu_be);
            ggml_backend_tensor_set(tw3_cpu, w_raw.data(), 0, static_cast<size_t>(N_VIEW) * 144U);

            struct ggml_context* ctx_g = ggml_init(p);
            struct ggml_cgraph* gf = ggml_new_graph(ctx_g);
            ggml_tensor* tx = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, K, M);
            ggml_set_input(tx);
            ggml_tensor* ty = ggml_mul_mat(ctx_g, tw3_cpu, tx);
            ggml_set_output(ty);
            ggml_build_forward_expand(gf, ty);

            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_be));
            CHECK_TRUE(ggml_gallocr_alloc_graph(galloc, gf), "cpu alloc for 3-row Q4_K ref");
            ggml_backend_tensor_set(tx, x_host.data(), 0, x_host.size() * sizeof(float));
            CHECK_TRUE(ggml_backend_graph_compute(cpu_be, gf) == GGML_STATUS_SUCCESS, "cpu compute 3-row Q4_K ref");
            ref_y.resize(static_cast<size_t>(M * N_VIEW), 0.0f);
            ggml_backend_tensor_get(ty, ref_y.data(), 0, ref_y.size() * sizeof(float));

            ggml_gallocr_free(galloc);
            ggml_free(ctx_g);
            ggml_backend_buffer_free(w_buf);
            ggml_free(ctx_w);
            ggml_backend_free(cpu_be);
        }
        double ref_energy = 0.0;
        for (float v : ref_y) ref_energy += static_cast<double>(v) * static_cast<double>(v);
        CHECK_TRUE(ref_energy > 1e-2, "3-row Q4_K CPU reference must be non-zero");

        std::fflush(stdout);
        std::fflush(stderr);
        const pid_t pid = fork();
        CHECK_TRUE(pid >= 0, "fork for 3-row BITCAST Q4_K slot-shape test");
        if (pid == 0) {
            setenv("CLOUDTPU_MOCK", "1", 1);
            setenv("CLOUDTPU_MOCK_DEVICES", "1", 1);
            setenv("CLOUDTPU_SPMD_CHIPS", "1", 1);
            setenv("CLOUDTPU_Q4_MODE", "bitcast", 1);
            setenv("CLOUDTPU_Q4K_BF16_SCALES", "0", 1);
            ggml_backend_t be = ggml_backend_cloudtpu_init();
            bool ok = false;
            if (be != nullptr) {
                const size_t mem = ggml_tensor_overhead() * 24 + ggml_graph_overhead() + 4096;
                struct ggml_init_params p = {mem, nullptr, true};
                struct ggml_context* ctx_w = ggml_init(p);
                ggml_tensor* tw4 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_K, K, N_FULL);
                ggml_backend_buffer_t w_buf = ggml_backend_alloc_ctx_tensors(ctx_w, be);
                // Populate bctx->slots[0] with the 4-row BITCAST (Q4_K_I4 + U8) slot
                ggml_backend_tensor_set(tw4, w_raw.data(), 0, w_raw.size());

                // Now create a 3-row Q4_K tensor at the exact same buffer offset 0 (different shape)
                struct ggml_context* ctx_g = ggml_init(p);
                struct ggml_cgraph* gf = ggml_new_graph(ctx_g);
                ggml_tensor* tx = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, K, M);
                ggml_set_input(tx);
                ggml_tensor* tw3 = ggml_new_tensor_2d(ctx_g, GGML_TYPE_Q4_K, K, N_VIEW);
                tw3->buffer = tw4->buffer;
                tw3->data = tw4->data;
                ggml_tensor* ty = ggml_mul_mat(ctx_g, tw3, tx);
                ggml_set_output(ty);
                ggml_build_forward_expand(gf, ty);

                ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (ggml_gallocr_alloc_graph(galloc, gf)) {
                    ggml_backend_tensor_set(tx, x_host.data(), 0, x_host.size() * sizeof(float));
                    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
                        std::vector<float> got(static_cast<size_t>(M * N_VIEW), 0.0f);
                        ggml_backend_tensor_get(ty, got.data(), 0, got.size() * sizeof(float));
                        ok = (compute_nmse(got, ref_y) < 5e-4);
                    }
                }
                ggml_gallocr_free(galloc);
                ggml_free(ctx_g);
                ggml_backend_buffer_free(w_buf);
                ggml_free(ctx_w);
                ggml_backend_free(be);
            }
            _exit(ok ? 0 : 1);
        }
        int status = 0;
        CHECK_TRUE(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                   "3-row Q4_K tensor aliasing 4-row BITCAST slot must slice weights and match CPU reference");
    }
    std::printf("  [PASS] ggml_cgraph reshaped host CPY bucket writeback & 3-row BITCAST Q4_K slot-shape fallback\n");
#endif
}
#endif

void test_freetile_mosaic_lowering(cloudtpu::PjrtRuntime& rt) {
    // 1) Q4_K_I4 S4 (M=1 pad8, K=512, N=256) with unshifted m: w = (s4 + 8)*d + m
    {
        const int64_t M = 1, N = 256, NB = 16, K = NB * 32;  // K=512
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<uint16_t> m_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));

        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t db = cloudtpu::f32_to_bf16(0.02f * static_cast<float>((n + b) % 7 + 1));
                const float d = cloudtpu::bf16_to_f32(db);
                const uint16_t mb = cloudtpu::f32_to_bf16(-0.08f + 0.015f * static_cast<float>((n * 2 + b) % 9));
                const float m = cloudtpu::bf16_to_f32(mb);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                m_bf16[static_cast<size_t>(b * N + n)] = mb;
                for (int64_t k = 0; k < 32; ++k) {
                    const int8_t s4 = static_cast<int8_t>(((n * 5 + b * 3 + k * 7) & 0x0F) - 8);
                    qs_s4[static_cast<size_t>((b * 32 + k) * N + n)] = s4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(s4 + 8) * d + m;
                }
            }
        }

        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.07f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += x[static_cast<size_t>(k)] * w_ref[static_cast<size_t>(n * K + k)];
            ref[static_cast<size_t>(n)] = acc;
        }

        auto bqs = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bm = rt.buffer_from_host(0, m_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
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
        CHECK_TRUE(prog.mlir_text.find("stablehlo.custom_call @tpu_custom_call") != std::string::npos,
                   "Q4_K_I4 S4 FreeTile Mosaic emits @tpu_custom_call");
        CHECK_TRUE(prog.mlir_text.find("stablehlo.dot_general") == std::string::npos,
                   "Q4_K_I4 S4 FreeTile Mosaic replaces dot_general");
        auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
        CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 S4 FreeTile Mosaic NMSE < 5e-4");

        // Verify invalid Mosaic dialect version fails compile cleanly & falls back to pure StableHLO
        cloudtpu::GraphProgramSpec bad_ver = spec;
        bad_ver.freetile_mosaic_version = 999;
        const auto bad_prog = cloudtpu::emit_stablehlo_program(bad_ver);
        CHECK_TRUE(bad_prog.cache_key != prog.cache_key, "Mosaic version tracked in cache_key");
        if (rt.is_mock()) {
            bool threw = false;
            try {
                (void)rt.compile_or_get(bad_prog.cache_key, bad_prog.mlir_text, 0, rt.make_spmd_compile_options(1));
            } catch (const std::exception&) {
                threw = true;
            }
            CHECK_TRUE(threw, "Invalid Mosaic dialect version rejects cleanly in MockPJRT");
        }
        cloudtpu::GraphProgramSpec fb_spec = bad_ver;
        fb_spec.freetile_mosaic = false;
        const auto fb_prog = cloudtpu::emit_stablehlo_program(fb_spec);
        CHECK_TRUE(fb_prog.mlir_text.find("@tpu_custom_call") == std::string::npos,
                   "Fallback program uses pure StableHLO");
        auto got_fb = run_emitted(rt, fb_prog, {bqs, bd, bm, bx});
        CHECK_TRUE(compute_nmse(got_fb, ref) < 5e-4, "Pure StableHLO fallback NMSE < 5e-4");
    }

    // 2) Q4_K_I4 U4 (M=2 pad8, K=256, N=128) with w = u4*d + m (no +8 shift)
    {
        const int64_t M = 2, N = 128, NB = 8, K = NB * 32;  // K=256
        std::vector<uint8_t> qs_u4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<uint16_t> m_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t db = cloudtpu::f32_to_bf16(0.018f * static_cast<float>((n + b * 2) % 7 + 1));
                const float d = cloudtpu::bf16_to_f32(db);
                const uint16_t mb = cloudtpu::f32_to_bf16(-0.10f + 0.02f * static_cast<float>((n + b) % 7));
                const float m = cloudtpu::bf16_to_f32(mb);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                m_bf16[static_cast<size_t>(b * N + n)] = mb;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 3 + b * 5 + k * 7) & 0x0F);
                    qs_u4[static_cast<size_t>((b * 32 + k) * N + n)] = u4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = static_cast<float>(u4) * d + m;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.04f);
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
        auto bqs = rt.buffer_from_host(0, qs_u4.data(), cloudtpu::PjrtDType::U4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bm = rt.buffer_from_host(0, m_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::U4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, N, K}};
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
        CHECK_TRUE(prog.mlir_text.find("stablehlo.custom_call @tpu_custom_call") != std::string::npos,
                   "Q4_K_I4 U4 FreeTile Mosaic emits @tpu_custom_call");
        auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
        CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_K_I4 U4 FreeTile Mosaic NMSE < 5e-4");

        // Verify MockPJRT rejects unsigned ui4 inside Mosaic MLIR body (MLIR requires signless i4)
        if (rt.is_mock()) {
            const std::string bad_mosaic =
                "module attributes {stable_mosaic.version = 17 : i64} { vector<256x128xui4> }";
            // Base64 encode bad_mosaic and inject into a minimal custom_call
            static constexpr char kB64[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string b64;
            for (size_t i = 0; i < bad_mosaic.size(); i += 3) {
                const uint32_t b0 = static_cast<unsigned char>(bad_mosaic[i]);
                const uint32_t b1 = (i + 1 < bad_mosaic.size()) ? static_cast<unsigned char>(bad_mosaic[i + 1]) : 0U;
                const uint32_t b2 = (i + 2 < bad_mosaic.size()) ? static_cast<unsigned char>(bad_mosaic[i + 2]) : 0U;
                const uint32_t t = (b0 << 16) | (b1 << 8) | b2;
                b64.push_back(kB64[(t >> 18) & 0x3F]);
                b64.push_back(kB64[(t >> 12) & 0x3F]);
                b64.push_back((i + 1 < bad_mosaic.size()) ? kB64[(t >> 6) & 0x3F] : '=');
                b64.push_back((i + 2 < bad_mosaic.size()) ? kB64[t & 0x3F] : '=');
            }
            const std::string bad_mlir =
                "module @bad_ui4 {\n"
                "  func.func @main(%arg0: tensor<8x256xbf16>, %arg1: tensor<256x128xui4>) -> tensor<8x128xf32> {\n"
                "    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) "
                "{backend_config = \"{\\\"custom_call_config\\\": {\\\"body\\\": \\\"" + b64 +
                "\\\", \\\"serialization_format\\\": 1}}\", kernel_name = \"ft_q4_k_bad\"} : "
                "(tensor<8x256xbf16>, tensor<256x128xui4>) -> tensor<8x128xf32>\n"
                "    return %0 : tensor<8x128xf32>\n"
                "  }\n"
                "}\n";
            bool threw = false;
            try {
                (void)rt.compile_or_get("bad_ui4_mosaic_test", bad_mlir, 0, rt.make_spmd_compile_options(1));
            } catch (const std::exception&) {
                threw = true;
            }
            CHECK_TRUE(threw, "MockPJRT rejects unsigned xui4 inside Mosaic MLIR body");
        }
    }

    // 3) Q4_0_I4 U4 (M=1 pad8, K=256, N=128) with w = (u4 - 8)*d
    {
        const int64_t M = 1, N = 128, NB = 8, K = NB * 32;  // K=256
        std::vector<uint8_t> qs_u4(static_cast<size_t>(NB * 32 * N));
        std::vector<uint16_t> d_f16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t dh = cloudtpu::f32_to_f16(0.025f * static_cast<float>((n + b) % 5 + 1));
                const float d = cloudtpu::f16_to_f32(dh);
                d_f16[static_cast<size_t>(b * N + n)] = dh;
                for (int64_t k = 0; k < 32; ++k) {
                    const uint8_t u4 = static_cast<uint8_t>((n * 7 + b * 3 + k * 5) & 0x0F);
                    qs_u4[static_cast<size_t>((b * 32 + k) * N + n)] = u4;
                    w_ref[static_cast<size_t>(n * K + b * 32 + k)] = (static_cast<float>(u4) - 8.0f) * d;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.09f);
        std::vector<float> ref(static_cast<size_t>(M * N), 0.0f);
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += x[static_cast<size_t>(k)] * w_ref[static_cast<size_t>(n * K + k)];
            ref[static_cast<size_t>(n)] = acc;
        }
        auto bqs = rt.buffer_from_host(0, qs_u4.data(), cloudtpu::PjrtDType::U4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_f16.data(), cloudtpu::PjrtDType::F16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::U4, cloudtpu::TensorStorageKind::Q4_0_I4, {1, 1, N, K}};
        spec.inputs.push_back(w_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, K}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, M, N}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.custom_call @tpu_custom_call") != std::string::npos,
                   "Q4_0_I4 U4 FreeTile Mosaic emits @tpu_custom_call");
        auto got = run_emitted(rt, prog, {bqs, bd, bx});
        CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q4_0_I4 U4 FreeTile Mosaic NMSE < 5e-4");
    }

    // 4) Q6_K_SPLIT (M=3 pad8, K=256, N=128) with FreeTile -> Mosaic tpu_custom_call
    {
        const int64_t M = 3, N = 128, NB = 16, K = NB * 16;  // K=256
        std::vector<int8_t> qs_s8(static_cast<size_t>(NB * 16 * N));
        std::vector<uint16_t> d_bf16(static_cast<size_t>(NB * N));
        std::vector<float> w_ref(static_cast<size_t>(N * K));
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t db = cloudtpu::f32_to_bf16(0.015f * static_cast<float>((n + b * 3) % 11 + 1));
                const float d = cloudtpu::bf16_to_f32(db);
                d_bf16[static_cast<size_t>(b * N + n)] = db;
                for (int64_t k = 0; k < 16; ++k) {
                    const int8_t q = static_cast<int8_t>(((n * 3 + b * 5 + k * 7) % 63) - 31);
                    qs_s8[static_cast<size_t>((b * 16 + k) * N + n)] = q;
                    w_ref[static_cast<size_t>(n * K + b * 16 + k)] = static_cast<float>(q) * d;
                }
            }
        }
        std::vector<float> x(static_cast<size_t>(M * K));
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(static_cast<float>(i) * 0.05f);
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

        auto bqs = rt.buffer_from_host(0, qs_s8.data(), cloudtpu::PjrtDType::S8, {1, 1, NB, 16, N});
        auto bd = rt.buffer_from_host(0, d_bf16.data(), cloudtpu::PjrtDType::BF16, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
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
        CHECK_TRUE(prog.mlir_text.find("stablehlo.custom_call @tpu_custom_call") != std::string::npos,
                   "Q6_K_SPLIT FreeTile Mosaic emits @tpu_custom_call");
        auto got = run_emitted(rt, prog, {bqs, bd, bx});
        CHECK_TRUE(compute_nmse(got, ref) < 5e-4, "Q6_K_SPLIT FreeTile Mosaic NMSE < 5e-4");
    }

    // 5) Unaligned shape (N=16) & non-Mosaic target (BUENOS_A0) cleanly fall back to pure StableHLO
    {
        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
        cloudtpu::TensorSpec w_spec{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, 16, 256}};
        w_spec.scale_dtype = cloudtpu::PjrtDType::BF16;
        spec.inputs.push_back(w_spec);
        spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 256}});
        cloudtpu::OpNodeSpec mm;
        mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        mm.src_ids = {1, 2};
        mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 16}};
        spec.ops.push_back(mm);
        spec.output_ids = {3};

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("@tpu_custom_call") == std::string::npos,
                   "Unaligned N=16 cleanly falls back to pure StableHLO");
        CHECK_TRUE(prog.mlir_text.find("stablehlo.dot_general") != std::string::npos,
                   "Unaligned fallback emits pure StableHLO dot_general");

        cloudtpu::GraphProgramSpec b0_spec;
        b0_spec.bf16_matmul = true;
        b0_spec.freetile_mosaic = true;
        b0_spec.freetile_chip_target = 3;  // BUENOS_A0 (supports_mosaic == false)
        cloudtpu::TensorSpec b0_w{1, cloudtpu::PjrtDType::S4, cloudtpu::TensorStorageKind::Q4_K_I4, {1, 1, 128, 256}};
        b0_w.scale_dtype = cloudtpu::PjrtDType::BF16;
        b0_spec.inputs.push_back(b0_w);
        b0_spec.inputs.push_back({2, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 256}});
        cloudtpu::OpNodeSpec b0_mm;
        b0_mm.kind = cloudtpu::CloudOpKind::MUL_MAT;
        b0_mm.src_ids = {1, 2};
        b0_mm.dst = {3, cloudtpu::PjrtDType::F32, cloudtpu::TensorStorageKind::DENSE, {1, 1, 1, 128}};
        b0_spec.ops.push_back(b0_mm);
        b0_spec.output_ids = {3};

        const auto b0_prog = cloudtpu::emit_stablehlo_program(b0_spec);
        CHECK_TRUE(b0_prog.mlir_text.find("@tpu_custom_call") == std::string::npos,
                   "BUENOS_A0 target cleanly skips Mosaic without throwing");
        CHECK_TRUE(b0_prog.mlir_text.find("stablehlo.dot_general") != std::string::npos,
                   "BUENOS_A0 fallback emits pure StableHLO dot_general");
    }

    // 6) Q4_K_I4 with F32 scales/mins (CLOUDTPU_Q4K_BF16_SCALES=0) preserves exact F32 cancellation:
    //    nibble q=7 (s4=-1), d=1027/1024, m=-7*d -> exact 0.0f (BF16 narrowing would yield -0.03125/elem!)
    {
        const int64_t M = 1, N = 128, NB = 8, K = NB * 32;  // K=256
        const float d_val = 1027.0f / 1024.0f;
        const float m_val = -7.0f * d_val;
        std::vector<int8_t> qs_s4(static_cast<size_t>(NB * 32 * N), static_cast<int8_t>(-1));  // s4=-1 => q=s4+8=7
        std::vector<float> d_f32(static_cast<size_t>(NB * N), d_val);
        std::vector<float> m_f32(static_cast<size_t>(NB * N), m_val);
        std::vector<float> x(static_cast<size_t>(M * K), 1.0f);

        auto bqs = rt.buffer_from_host(0, qs_s4.data(), cloudtpu::PjrtDType::S4, {1, 1, NB, 32, N});
        auto bd = rt.buffer_from_host(0, d_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bm = rt.buffer_from_host(0, m_f32.data(), cloudtpu::PjrtDType::F32, {1, 1, NB, N});
        auto bx = rt.buffer_from_host(0, x.data(), cloudtpu::PjrtDType::F32, {1, 1, M, K});

        cloudtpu::GraphProgramSpec spec;
        spec.bf16_matmul = true;
        spec.freetile_mosaic = true;
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

        const auto prog = cloudtpu::emit_stablehlo_program(spec);
        CHECK_TRUE(prog.mlir_text.find("stablehlo.custom_call @tpu_custom_call") != std::string::npos,
                   "F32-scale Q4_K_I4 emits @tpu_custom_call");
        CHECK_TRUE(prog.mlir_text.find("tensor<8x128xf32>, tensor<8x128xf32>) -> tensor<8x128xf32>") != std::string::npos,
                   "F32-scale Q4_K_I4 passes f32 d and m directly to @tpu_custom_call without BF16 narrowing");
        auto got = run_emitted(rt, prog, {bqs, bd, bm, bx});
        CHECK_TRUE(got.size() == static_cast<size_t>(N), "F32-scale cancellation output length");
        for (float v : got) {
            CHECK_NEAR(v, 0.0f, 1e-6f, "F32-scale Q4_K_I4 exact cancellation q=7, d=1027/1024, m=-7*d yields 0");
        }
    }

    std::printf("  [PASS] FreeTile Mosaic lowering (Q4_K_I4 S4/U4/F32-scales, Q4_0_I4 U4, Q6_K_SPLIT, fallback & dialect guard)\n");
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
    test_subbyte_s4_u4_packing_and_hbm(rt);
    test_q4_i4_single_read_matmul_and_bitcast(rt);
    test_q4_i4_get_rows_coexistence(rt);
    test_q4_i4_spmd_sharding();
    test_bf16_scales_zero_f32_dequant_and_pad_m(rt);
    test_one_graph_multilayer_decode_and_donation();
    test_one_graph_spmd_kv_donation(4);
    test_continuous_batching_padded_buckets();
    test_freetile_mosaic_lowering(rt);
#ifdef CLOUDTPU_HAS_GGML
    test_ggml_cgraph_odd_s4_shard_n();
    test_ggml_cgraph_reshaped_host_cpy_and_q4_bitcast_view();
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
