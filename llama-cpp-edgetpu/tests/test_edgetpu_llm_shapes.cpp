// Dedicated verification suite for ggml-edgetpu (Pixel Tensor G5 / buenos_a0):
//   1. LLM-shape MUL_MAT accuracy sweep vs CPU (M=1,4,16; K=64..2048; N=64..2048; Q8_0 & F32)
//   2. EDGETPU buffer upload/readback & weight repacking verification
//   3. Hybrid ggml_backend_sched execution verifying MUL_MAT -> EDGETPU and glue ops -> CPU
//   4. OPENBUENOS_MIN_ELEMENTS threshold verification

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-edgetpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double compute_nmse(const float * ref, const float * got, size_t n, float * out_max_abs) {
    double mse = 0.0;
    double ref_var = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = got[i] - ref[i];
        const float ad   = std::fabs(diff);
        if (ad > max_abs) {
            max_abs = ad;
        }
        mse     += static_cast<double>(diff) * static_cast<double>(diff);
        ref_var += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    if (out_max_abs) {
        *out_max_abs = max_abs;
    }
    return (ref_var > 0.0) ? (mse / ref_var) : mse;
}

static bool test_mul_mat_shape(ggml_backend_t b_tpu,
                               ggml_backend_t b_cpu,
                               ggml_type      type_a,
                               ggml_type      type_b,
                               int64_t        M,
                               int64_t        K,
                               int64_t        N,
                               double         max_allowed_nmse) {
    std::mt19937 rng(static_cast<uint32_t>(M * 1009 + K * 37 + N * 13 + static_cast<int>(type_a)));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_f32(static_cast<size_t>(N) * K);
    std::vector<float> b_f32(static_cast<size_t>(M) * K);
    for (float & v : a_f32) v = dist(rng);
    for (float & v : b_f32) v = dist(rng);

    std::vector<uint8_t> a_raw(ggml_row_size(type_a, K) * static_cast<size_t>(N));
    std::vector<uint8_t> b_raw(ggml_row_size(type_b, K) * static_cast<size_t>(M));

    if (type_a == GGML_TYPE_F32) {
        std::memcpy(a_raw.data(), a_f32.data(), a_raw.size());
    } else {
        ggml_quantize_chunk(type_a, a_f32.data(), a_raw.data(), 0, N, K, nullptr);
    }
    if (type_b == GGML_TYPE_F32) {
        std::memcpy(b_raw.data(), b_f32.data(), b_raw.size());
    } else {
        ggml_quantize_chunk(type_b, b_f32.data(), b_raw.data(), 0, M, K, nullptr);
    }

    auto run_on_backend = [&](ggml_backend_t backend, bool is_tpu, std::vector<float> & out) -> bool {
        const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        ggml_init_params params { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor * ta  = ggml_new_tensor_2d(ctx, type_a, K, N);
        ggml_tensor * tb  = ggml_new_tensor_2d(ctx, type_b, K, M);
        ggml_tensor * dst = ggml_mul_mat(ctx, ta, tb);

        if (is_tpu && !ggml_backend_supports_op(backend, dst)) {
            std::fprintf(stderr, "FAIL: EDGETPU rejected supported shape (%s x %s) M=%lld K=%lld N=%lld\n",
                         ggml_type_name(type_a), ggml_type_name(type_b),
                         (long long)M, (long long)K, (long long)N);
            ggml_free(ctx);
            return false;
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            ggml_free(ctx);
            return false;
        }
        if (is_tpu) {
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ggml_backend_tensor_set(ta, a_raw.data(), 0, a_raw.size());
        ggml_backend_tensor_set(tb, b_raw.data(), 0, b_raw.size());

        // Verify bit-exact weight readback from EDGETPU buffer.
        if (is_tpu && type_a == GGML_TYPE_Q8_0) {
            std::vector<uint8_t> readback(a_raw.size());
            ggml_backend_tensor_get(ta, readback.data(), 0, readback.size());
            if (std::memcmp(readback.data(), a_raw.data(), a_raw.size()) != 0) {
                std::fprintf(stderr, "FAIL: EDGETPU Q8_0 buffer readback mismatch!\n");
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                return false;
            }
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }

        out.resize(static_cast<size_t>(M) * N);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    std::vector<float> ref, got;
    if (!run_on_backend(b_cpu, false, ref)) return false;
    if (!run_on_backend(b_tpu, true, got)) return false;

    float max_abs = 0.0f;
    const double nmse = compute_nmse(ref.data(), got.data(), ref.size(), &max_abs);
    const bool ok = (nmse <= max_allowed_nmse);
    std::printf("  [%s] MUL_MAT %-4s x %-4s  M=%-2lld K=%-4lld N=%-4lld  nmse=%.3e  max_abs=%.4f\n",
                ok ? "PASS" : "FAIL",
                ggml_type_name(type_a), ggml_type_name(type_b),
                (long long)M, (long long)K, (long long)N,
                nmse, max_abs);
    return ok;
}

static bool test_hybrid_scheduler_routing(ggml_backend_t b_tpu, ggml_backend_t b_cpu) {
    std::printf("\n[Hybrid Scheduler Routing Check]\n");
    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ggml_init_params params { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    const int64_t K = 256, N = 256, M = 4;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, N);

    ggml_tensor * norm = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_tensor * mm   = ggml_mul_mat(ctx, w, norm);
    ggml_tensor * sm   = ggml_soft_max(ctx, mm);

    // Verify supports_op contract explicitly.
    if (ggml_backend_supports_op(b_tpu, norm)) {
        std::fprintf(stderr, "FAIL: EDGETPU should NOT support RMS_NORM\n");
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_supports_op(b_tpu, mm)) {
        std::fprintf(stderr, "FAIL: EDGETPU MUST support Q8_0 x F32 MUL_MAT\n");
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_supports_op(b_tpu, sm)) {
        std::fprintf(stderr, "FAIL: EDGETPU should NOT support SOFT_MAX\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_t backends[2] = { b_tpu, b_cpu };
    ggml_backend_buffer_type_t bufts[2] = {
        ggml_backend_edgetpu_buffer_type(),
        ggml_backend_cpu_buffer_type(),
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 1024, false, true);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_edgetpu_buffer_type());
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> x_vals(M * K, 0.25f);
    std::vector<float> w_vals(N * K, 0.1f);
    std::vector<uint8_t> w_q8(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w_vals.data(), w_q8.data(), 0, N, K, nullptr);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sm);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "FAIL: ggml_backend_sched_alloc_graph failed\n");
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(x, x_vals.data(), 0, x_vals.size() * sizeof(float));
    ggml_backend_tensor_set(w, w_q8.data(), 0, w_q8.size());

    ggml_backend_t b_norm = ggml_backend_sched_get_tensor_backend(sched, norm);
    ggml_backend_t b_mm   = ggml_backend_sched_get_tensor_backend(sched, mm);
    ggml_backend_t b_sm   = ggml_backend_sched_get_tensor_backend(sched, sm);

    std::printf("  RMS_NORM -> %s\n", b_norm ? ggml_backend_name(b_norm) : "NULL");
    std::printf("  MUL_MAT  -> %s\n", b_mm ? ggml_backend_name(b_mm) : "NULL");
    std::printf("  SOFT_MAX -> %s\n", b_sm ? ggml_backend_name(b_sm) : "NULL");

    const bool routing_ok = (b_norm == b_cpu) && (b_mm == b_tpu) && (b_sm == b_cpu);
    const ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    std::vector<float> out(M * N);
    ggml_backend_tensor_get(sm, out.data(), 0, out.size() * sizeof(float));

    // Each softmax row over N=256 equal logits should sum to 1.0 (~1/256 each).
    float row0_sum = 0.0f;
    for (int64_t n = 0; n < N; ++n) row0_sum += out[n];
    const bool val_ok = (st == GGML_STATUS_SUCCESS) && (std::fabs(row0_sum - 1.0f) < 1e-4f);
    std::printf("  Hybrid compute status=%d, softmax row0 sum=%.6f -> %s\n",
                (int)st, row0_sum, (routing_ok && val_ok) ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    return routing_ok && val_ok;
}

static bool test_zero_repack_and_auto_routing(ggml_backend_t b_tpu, ggml_backend_t b_cpu) {
    std::printf("\n[Zero-Repack Prepack, Weight Mutation Invalidation, & Auto-Routing Gates]\n");
    bool ok = true;

    // 1. Verify MAX_N=65536 gate keeps 200k vocab head on CPU and allows <= 65536 on EDGETPU;
    //    and MUL_MAT_ID auto (>64 MiB) keeps huge MoE on CPU while allowing small MoE.
    {
        const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        ggml_init_params params { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor * x_voc   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  64, 1);
        ggml_tensor * w_big_n = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 70000);
        ggml_tensor * mm_big  = ggml_mul_mat(ctx, w_big_n, x_voc);

        if (ggml_backend_supports_op(b_tpu, mm_big)) {
            std::fprintf(stderr, "FAIL: EDGETPU should reject N=70000 (> default MAX_N=65536)\n");
            ok = false;
        }

        ggml_tensor * w_huge_moe = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 2304, 2560, 64);
        ggml_tensor * x_moe      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  2304, 1, 1);
        ggml_tensor * ids_moe    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  8, 1);
        ggml_tensor * mmid_huge  = ggml_mul_mat_id(ctx, w_huge_moe, x_moe, ids_moe);
        if (ggml_backend_supports_op(b_tpu, mmid_huge)) {
            std::fprintf(stderr, "FAIL: EDGETPU auto gate should reject 360 MiB MoE MUL_MAT_ID\n");
            ok = false;
        }

        ggml_tensor * w_small_moe = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 64, 64, 8);
        ggml_tensor * x_small     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  64, 1, 1);
        ggml_tensor * ids_small   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  2, 1);
        ggml_tensor * mmid_small  = ggml_mul_mat_id(ctx, w_small_moe, x_small, ids_small);
        if (!ggml_backend_supports_op(b_tpu, mmid_small)) {
            std::fprintf(stderr, "FAIL: EDGETPU should allow 32 KiB small MoE MUL_MAT_ID\n");
            ok = false;
        }

        ggml_free(ctx);
    }

    // 2. Sibling [Q, K, V] Q4_0 graph on USAGE_WEIGHTS EDGETPU buffer + mutation invalidation.
    {
        constexpr int64_t M = 1, K = 256, Nq = 256, Nk = 128;
        const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        ggml_init_params params { ctx_size, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;

        ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, Nq);
        ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, Nk);
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_edgetpu_buffer_type());
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        ggml_tensor * mq = ggml_mul_mat(ctx, wq, x);
        ggml_tensor * mk = ggml_mul_mat(ctx, wk, x);
        ggml_tensor * out_sum = ggml_add(ctx, ggml_repeat(ctx, mk, mq), mq);

        ggml_backend_t backends[2] = { b_tpu, b_cpu };
        ggml_backend_buffer_type_t bufts[2] = {
            ggml_backend_edgetpu_buffer_type(),
            ggml_backend_cpu_buffer_type(),
        };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 1024, false, true);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_sum);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            std::fprintf(stderr, "FAIL: alloc_graph failed in sibling prepack test\n");
            ggml_backend_sched_free(sched);
            ggml_backend_buffer_free(buf_w);
            ggml_free(ctx);
            return false;
        }

        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> x_vals(M * K, 0.5f);
        std::vector<float> wq_f1(Nq * K), wq_f2(Nq * K), wk_f(Nk * K);
        for (float & v : wq_f1) v = dist(rng);
        for (float & v : wq_f2) v = 1.0f;
        for (float & v : wk_f)  v = dist(rng);

        std::vector<uint8_t> wq_q4(ggml_nbytes(wq)), wk_q4(ggml_nbytes(wk));
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wq_f1.data(), wq_q4.data(), 0, Nq, K, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wk_f.data(),  wk_q4.data(), 0, Nk, K, nullptr);

        ggml_backend_tensor_set(wq, wq_q4.data(), 0, wq_q4.size());
        ggml_backend_tensor_set(wk, wk_q4.data(), 0, wk_q4.size());
        ggml_backend_tensor_set(x,  x_vals.data(), 0, x_vals.size() * sizeof(float));

        ok &= (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        std::vector<float> y1(M * Nq), y2(M * Nq), y3(M * Nq);
        ggml_backend_tensor_get(out_sum, y1.data(), 0, y1.size() * sizeof(float));

        // Second compute with same weights & input -> bit-exact match.
        ok &= (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out_sum, y2.data(), 0, y2.size() * sizeof(float));
        for (size_t i = 0; i < y1.size(); ++i) {
            if (std::fabs(y1[i] - y2[i]) > 1e-6f) {
                ok = false;
                break;
            }
        }

        // Mutate wq via buffer_set -> version bumps -> next eval MUST reflect updated weights.
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wq_f2.data(), wq_q4.data(), 0, Nq, K, nullptr);
        ggml_backend_tensor_set(wq, wq_q4.data(), 0, wq_q4.size());
        ok &= (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out_sum, y3.data(), 0, y3.size() * sizeof(float));

        double diff_sum = 0.0;
        for (size_t i = 0; i < y1.size(); ++i) diff_sum += std::fabs(y3[i] - y1[i]);
        if (diff_sum <= 1.0) {
            std::fprintf(stderr, "FAIL: mutated EDGETPU weight buffer did not invalidate cached weights\n");
            ok = false;
        }

        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx);
    }

    std::printf("  Zero-repack & auto-routing gates -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("EDGETPU");
    if (!dev) {
        std::fprintf(stderr, "FAIL: EDGETPU device not registered in ggml backend registry!\n");
        return 1;
    }
    std::printf("Registered device: name=\"%s\" desc=\"%s\" type=%d\n",
                ggml_backend_dev_name(dev),
                ggml_backend_dev_description(dev),
                static_cast<int>(ggml_backend_dev_type(dev)));

    ggml_backend_t b_tpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t b_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!b_tpu || !b_cpu) {
        std::fprintf(stderr, "FAIL: failed to initialize EDGETPU or CPU backend\n");
        return 1;
    }
    std::printf("EDGETPU initialized (mock_mode=%d)\n\n", ggml_backend_edgetpu_is_mock(b_tpu) ? 1 : 0);

    bool all_ok = true;
    struct Shape { int64_t M; int64_t K; int64_t N; };
    const Shape shapes[] = {
        {  1,   64,   64 },
        {  1,  256,  256 },
        {  1,  512, 1024 },
        {  1, 1024, 2048 },
        {  1, 2048, 2048 },
        {  4,   64,  256 },
        {  4,  512,  512 },
        {  4, 1024, 1024 },
        {  4, 2048, 2048 },
        { 16,   64,   64 },
        { 16,  512, 1024 },
        { 16, 2048, 2048 },
    };

    std::printf("[LLM Shape Accuracy Sweep: Q8_0 x F32]\n");
    for (const auto & s : shapes) {
        all_ok &= test_mul_mat_shape(b_tpu, b_cpu, GGML_TYPE_Q8_0, GGML_TYPE_F32, s.M, s.K, s.N, 5e-4);
    }

    std::printf("\n[LLM Shape Accuracy Sweep: Q8_0 x Q8_0]\n");
    all_ok &= test_mul_mat_shape(b_tpu, b_cpu, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1, 256, 256, 5e-4);
    all_ok &= test_mul_mat_shape(b_tpu, b_cpu, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 1024, 1024, 5e-4);

    std::printf("\n[LLM Shape Accuracy Sweep: F32 x F32 (dynamic INT8 quantize -> buenos INT8 GEMM -> F32)]\n");
    for (const auto & s : shapes) {
        all_ok &= test_mul_mat_shape(b_tpu, b_cpu, GGML_TYPE_F32, GGML_TYPE_F32, s.M, s.K, s.N, 5e-4);
    }

    all_ok &= test_hybrid_scheduler_routing(b_tpu, b_cpu);
    all_ok &= test_zero_repack_and_auto_routing(b_tpu, b_cpu);

    ggml_backend_free(b_tpu);
    ggml_backend_free(b_cpu);

    std::printf("\nOVERALL: %s\n", all_ok ? "ALL PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
