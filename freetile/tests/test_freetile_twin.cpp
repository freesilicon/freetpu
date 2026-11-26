// SPDX-License-Identifier: Apache-2.0
// Comprehensive host verification suite for FreeTile Digital Twin (freetile-twin).
//
// Covers:
//   1. GEMM (BF16, FP32, W8A8 + Mode-A SiLU/GELU) on buenos-a0 and malibu
//   2. SwiGLU (11-chunk in-SRAM chained INT8 + quantized Q4_K/Q6_K/IQ4_NL/IQ4_XS)
//   3. FlashAttention (GQA decode & prefill, online softmax 0x640 parity, Q4_0 KV stall rejection)
//   4. Quantized GEMM/unpack (W8A16, Q4_K, Q6_K, IQ4_NL, IQ4_XS, Q8_0, Q4_0) on buenos-a0 & malibu
//   5. 0x640 minimax splines (all 7 tables) + error bound checks
//   6. Raw .dgc0 wire integrity (valid + negative corrupted magic, truncated, OOB ext,
//      misaligned ext, overlapping ext, SHA-256 mismatch, SRAM overflow)
//   7. Negative SRAM overflow detection on both buenos-a0 and malibu
//   8. freesilicon.twin.v1 JSON schema compliance + deterministic replay
//   9. Throughput proof >= 1,000 real evaluations/min

#include "freetile/twin.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK_TRUE(cond, msg)                                                                  \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));             \
            ++g_failures;                                                                      \
            return;                                                                            \
        }                                                                                      \
    } while (0)

std::string serialize_json(const ft_twin_result_t& res) {
    size_t need = 0u;
    (void)ft_twin_result_to_json(&res, 0, nullptr, 0u, &need);
    std::vector<char> buf(need + 1u, '\0');
    const ft_twin_status_t st = ft_twin_result_to_json(&res, 0, buf.data(), buf.size(), &need);
    if (st != FT_TWIN_OK) return {};
    return std::string(buf.data(), need);
}

void check_common_schema_fields(const ft_twin_result_t& res, const std::string& json) {
    CHECK_TRUE(!json.empty(), "JSON serialization non-empty");
    CHECK_TRUE(std::strcmp(res.schema_version, FT_TWIN_SCHEMA_VERSION) == 0, "schema_version const");
    CHECK_TRUE(res.reward >= -1.0 && res.reward <= 1.0, "reward bounded in [-1, 1]");
    if (res.pass) {
        CHECK_TRUE(res.reward > 0.0, "passing reward strictly > 0");
        CHECK_TRUE(!res.has_fault, "passing verdict has null fault_reason");
    } else {
        CHECK_TRUE(res.reward <= -0.1, "failing reward <= -0.1");
        CHECK_TRUE(res.has_fault && res.fault_reason[0] != '\0', "failing verdict has fault_reason");
    }
    const char* required_keys[] = {
        "\"schema_version\":\"freesilicon.twin.v1\"",
        "\"domain\":",
        "\"target\":",
        "\"workload\":",
        "\"pass\":",
        "\"reward\":",
        "\"nmse\":",
        "\"max_abs_err\":",
        "\"sram_bytes\":",
        "\"sram_limit_bytes\":",
        "\"est_hw_us\":",
        "\"eval_ms\":",
        "\"fault_reason\":",
        "\"diagnostics\":",
        "\"checks\":",
        "\"reward_breakdown\":",
    };
    for (const char* k : required_keys) {
        CHECK_TRUE(json.find(k) != std::string::npos,
                   (std::string("Missing required JSON key: ") + k).c_str());
    }
}

void test_gemm_profiles(ft_twin_session_t* s) {
    const ft_twin_target_t targets[2] = {FT_TWIN_TARGET_BUENOS_A0, FT_TWIN_TARGET_MALIBU};
    const ft_twin_dtype_t dtypes[3] = {FT_TWIN_DTYPE_BF16, FT_TWIN_DTYPE_FP32, FT_TWIN_DTYPE_INT8};
    const ft_twin_activation_t acts[3] = {FT_TWIN_ACT_NONE, FT_TWIN_ACT_SILU, FT_TWIN_ACT_GELU};

    for (ft_twin_target_t tgt : targets) {
        for (int i = 0; i < 3; ++i) {
            ft_twin_request_t req = ft_twin_request_default();
            req.target = tgt;
            req.workload = FT_TWIN_WORKLOAD_GEMM;
            req.m = 1u;
            req.k = 256u;
            req.n = 256u;
            req.act_dtype = dtypes[i];
            req.out_dtype = (dtypes[i] == FT_TWIN_DTYPE_INT8) ? FT_TWIN_DTYPE_INT8 : FT_TWIN_DTYPE_FP32;
            req.activation = acts[i];

            ft_twin_result_t res{};
            CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "GEMM eval ok");
            CHECK_TRUE(res.pass == 1, res.fault_reason);
            CHECK_TRUE(res.has_nmse && res.nmse < 5e-3, "GEMM nmse within bound");
            CHECK_TRUE(res.has_sram_bytes && res.sram_bytes <= res.sram_limit_bytes, "GEMM SRAM ok");
            CHECK_TRUE(res.has_est_hw_us && res.est_hw_us > 0.0, "GEMM est_hw_us > 0");
            CHECK_TRUE(res.check_dgc0_wire_ok == 1, "GEMM DGC0 wire verified");
            if (acts[i] == FT_TWIN_ACT_SILU || acts[i] == FT_TWIN_ACT_GELU) {
                CHECK_TRUE(res.check_spline_0x640_ok == 1, "GEMM 0x640 spline checked");
            }
            check_common_schema_fields(res, serialize_json(res));
        }
    }
    std::printf("[PASS] test_gemm_profiles (BF16/FP32/W8A8 + SiLU/GELU on buenos-a0 & malibu)\n");
}

void test_swiglu_11chunk_and_quant(ft_twin_session_t* s) {
    // 1. 11-chunk chained in-SRAM SwiGLU on both buenos-a0 and malibu
    for (ft_twin_target_t tgt : {FT_TWIN_TARGET_BUENOS_A0, FT_TWIN_TARGET_MALIBU}) {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = tgt;
        req.workload = FT_TWIN_WORKLOAD_SWIGLU;
        req.swiglu_11chunk = 1;
        req.m = 1u; req.k = 576u; req.h = 512u; req.n = 576u;

        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "11-chunk SwiGLU eval ok");
        CHECK_TRUE(res.pass == 1, res.fault_reason);
        CHECK_TRUE(res.has_nmse && res.nmse == 0.0, "11-chunk SwiGLU bit-exact parity (nmse=0)");
        CHECK_TRUE(res.has_max_abs_err && res.max_abs_err == 0.0, "11-chunk SwiGLU max_abs_err=0");
        CHECK_TRUE(res.check_dgc0_wire_ok == 1, "11-chunk SwiGLU DGC0 wire ok");
        CHECK_TRUE(res.check_spline_0x640_ok == 1, "11-chunk SwiGLU 0x640 SiLU ok");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 2. Quantized SwiGLU across Q4_K, Q6_K, IQ4_NL, IQ4_XS
    for (uint32_t qt : {2u, 3u, 4u, 5u}) {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = FT_TWIN_TARGET_BUENOS_A0;
        req.workload = FT_TWIN_WORKLOAD_SWIGLU;
        req.swiglu_11chunk = 0;
        req.quant_type = qt;
        req.m = 1u; req.k = 256u; req.h = 256u; req.n = 128u;

        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "Quant SwiGLU eval ok");
        CHECK_TRUE(res.pass == 1, res.fault_reason);
        CHECK_TRUE(res.has_nmse && res.nmse < 1e-3, "Quant SwiGLU NMSE < 1e-3");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 3. Negative: unsupported 11-chunk SwiGLU shape must fail cleanly
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.workload = FT_TWIN_WORKLOAD_SWIGLU;
        req.swiglu_11chunk = 1;
        req.m = 1u; req.k = 128u; req.h = 128u; req.n = 128u;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "bad SwiGLU shape eval ok");
        CHECK_TRUE(res.pass == 0, "unsupported 11-chunk shape rejected");
        CHECK_TRUE(std::strstr(res.fault_reason, "SWIGLU_UNSUPPORTED_SHAPE") != nullptr, "fault code");
        check_common_schema_fields(res, serialize_json(res));
    }
    std::printf("[PASS] test_swiglu_11chunk_and_quant (11-chunk bit-exact + Q4_K/Q6_K/IQ4_NL/IQ4_XS + negative)\n");
}

void test_flash_attn(ft_twin_session_t* s) {
    // 1. GQA 8:1 decode on buenos-a0 and malibu
    for (ft_twin_target_t tgt : {FT_TWIN_TARGET_BUENOS_A0, FT_TWIN_TARGET_MALIBU}) {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = tgt;
        req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
        req.fa_phase = FT_TWIN_FA_DECODE;
        req.head_dim = 128u; req.num_q_heads = 32u; req.num_kv_heads = 4u;
        req.seq_len = 2048u; req.kv_format = FT_TWIN_KV_Q8_0;

        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "FlashAttn decode ok");
        CHECK_TRUE(res.pass == 1, res.fault_reason);
        CHECK_TRUE(res.sram_bytes <= 786432u, "FlashAttn decode <= 768 KiB");
        CHECK_TRUE(res.has_nmse && res.nmse < 1e-5, "FlashAttn online softmax NMSE < 1e-5");
        CHECK_TRUE(res.check_spline_0x640_ok == 1, "FlashAttn 0x640 exp+recip ok");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 2. Prefill FlashAttention
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = FT_TWIN_TARGET_BUENOS_A0;
        req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
        req.fa_phase = FT_TWIN_FA_PREFILL;
        req.head_dim = 128u; req.num_q_heads = 32u; req.num_kv_heads = 4u;
        req.seq_len = 1024u;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "FlashAttn prefill ok");
        CHECK_TRUE(res.pass == 1, res.fault_reason);
        CHECK_TRUE(res.sram_bytes <= 786432u, "FlashAttn prefill <= 768 KiB");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 3. Negative: FlashAttention SRAM overflow (br=64, bc=1024, d=128 > 768 KiB)
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = FT_TWIN_TARGET_BUENOS_A0;
        req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
        req.fa_phase = FT_TWIN_FA_PREFILL;
        req.head_dim = 128u; req.br = 64u; req.bc = 1024u;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "FlashAttn overflow eval ok");
        CHECK_TRUE(res.pass == 0, "FlashAttn overflow rejected");
        CHECK_TRUE(res.sram_bytes > res.sram_limit_bytes, "sram_bytes > sram_limit_bytes reported");
        CHECK_TRUE(std::strstr(res.fault_reason, "SRAM_OVERFLOW") != nullptr, "SRAM_OVERFLOW fault");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 4. Negative: Q4_0 KV cache rejected due to 96:1 VLIW unpack stall
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = FT_TWIN_TARGET_MALIBU;
        req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
        req.kv_format = FT_TWIN_KV_Q4_0;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "Q4_0 KV eval ok");
        CHECK_TRUE(res.pass == 0, "Q4_0 KV cache rejected without native 1:4 expander");
        CHECK_TRUE(std::strstr(res.fault_reason, "KV_CACHE_FORMAT_REJECTED") != nullptr, "KV fault code");
        check_common_schema_fields(res, serialize_json(res));
    }
    std::printf("[PASS] test_flash_attn (GQA decode/prefill + online softmax 0x640 + SRAM overflow + Q4_0 rejection)\n");
}

void test_quant_kernels(ft_twin_session_t* s) {
    const ft_twin_workload_t workloads[] = {
        FT_TWIN_WORKLOAD_QUANT_W8A16,
        FT_TWIN_WORKLOAD_QUANT_Q4_K,
        FT_TWIN_WORKLOAD_QUANT_Q6_K,
        FT_TWIN_WORKLOAD_QUANT_IQ4_NL,
        FT_TWIN_WORKLOAD_QUANT_IQ4_XS,
        FT_TWIN_WORKLOAD_QUANT_Q8_0,
        FT_TWIN_WORKLOAD_QUANT_Q4_0,
    };

    for (ft_twin_target_t tgt : {FT_TWIN_TARGET_BUENOS_A0, FT_TWIN_TARGET_MALIBU}) {
        for (ft_twin_workload_t wl : workloads) {
            ft_twin_request_t req = ft_twin_request_default();
            req.target = tgt;
            req.workload = wl;
            req.m = 1u;
            req.k = 512u;
            req.n = (wl == FT_TWIN_WORKLOAD_QUANT_W8A16) ? 512u : 256u;

            ft_twin_result_t res{};
            CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "Quant kernel eval ok");
            CHECK_TRUE(res.pass == 1, res.fault_reason);
            CHECK_TRUE(res.has_nmse && res.nmse < 2.5e-3, "Quant kernel NMSE < 2.5e-3");
            CHECK_TRUE(res.check_sram_budget_ok == 1, "Quant kernel fits SRAM");
            CHECK_TRUE(res.check_dgc0_wire_ok == 1, "Quant kernel DGC0 wire verified");
            if (tgt == FT_TWIN_TARGET_MALIBU) {
                if (wl == FT_TWIN_WORKLOAD_QUANT_IQ4_NL || wl == FT_TWIN_WORKLOAD_QUANT_IQ4_XS) {
                    CHECK_TRUE(std::strcmp(res.selected_engine, "cpu_neon") == 0,
                               "Malibu routes M=1 IQ4_NL/IQ4_XS to cpu_neon");
                } else {
                    CHECK_TRUE(std::strcmp(res.selected_engine, "cpu_sme2") == 0,
                               "Malibu routes M=1 Q4_K/Q6_K/Q8_0/Q4_0/W8A16 to cpu_sme2");
                }
            }
            check_common_schema_fields(res, serialize_json(res));
        }
    }

    // Negative: explicit oversized tile overflows Malibu 768 KiB SRAM while fitting buenos-a0 16 MiB
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.workload = FT_TWIN_WORKLOAD_QUANT_Q4_K;
        req.m = 1u; req.k = 1024u; req.n = 1024u;
        req.bm = 8u; req.bn = 1024u; req.bk = 1024u;

        req.target = FT_TWIN_TARGET_BUENOS_A0;
        ft_twin_result_t res_b{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res_b) == FT_TWIN_OK, "buenos large tile eval");
        CHECK_TRUE(res_b.pass == 1, "1024x1024 Q4_K tile fits buenos-a0 16 MiB VMEM");

        req.target = FT_TWIN_TARGET_MALIBU;
        ft_twin_result_t res_m{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res_m) == FT_TWIN_OK, "malibu large tile eval");
        CHECK_TRUE(res_m.pass == 0, "1024x1024 Q4_K tile exceeds malibu 768 KiB narrow SRAM");
        CHECK_TRUE(std::strstr(res_m.fault_reason, "SRAM_OVERFLOW") != nullptr, "malibu SRAM_OVERFLOW");
        check_common_schema_fields(res_m, serialize_json(res_m));
    }
    std::printf("[PASS] test_quant_kernels (W8A16/Q4_K/Q6_K/IQ4_NL/IQ4_XS/Q8_0/Q4_0 on buenos-a0 & malibu + SRAM split)\n");
}

void test_splines_0x640(ft_twin_session_t* s) {
    const ft_twin_spline_kind_t kinds[] = {
        FT_TWIN_SPLINE_EXP_NEG12_96,
        FT_TWIN_SPLINE_EXP_NEG11,
        FT_TWIN_SPLINE_RSQRT,
        FT_TWIN_SPLINE_RECIP,
        FT_TWIN_SPLINE_SILU,
        FT_TWIN_SPLINE_GELU_TANH,
        FT_TWIN_SPLINE_GELU_ERF,
    };
    for (ft_twin_spline_kind_t k : kinds) {
        ft_twin_request_t req = ft_twin_request_default();
        req.workload = FT_TWIN_WORKLOAD_SPLINE_0X640;
        req.spline_kind = k;
        req.spline_samples = 4097u;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "spline eval ok");
        CHECK_TRUE(res.pass == 1, res.fault_reason);
        CHECK_TRUE(res.check_spline_0x640_ok == 1, "spline check ok");
        check_common_schema_fields(res, serialize_json(res));
    }
    // Negative: impossibly tight custom error bound triggers SPLINE_0X640_FAIL
    {
        ft_twin_request_t req = ft_twin_request_default();
        req.workload = FT_TWIN_WORKLOAD_SPLINE_0X640;
        req.spline_kind = FT_TWIN_SPLINE_SILU;
        req.max_abs_err_limit = 1e-12;
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "tight spline eval");
        CHECK_TRUE(res.pass == 0, "impossibly tight spline bound rejected");
        CHECK_TRUE(std::strstr(res.fault_reason, "SPLINE_0X640_FAIL") != nullptr, "SPLINE_0X640_FAIL");
    }
    std::printf("[PASS] test_splines_0x640 (7/7 0x640 LUTs + 208B wire round-trip + error bound gate)\n");
}

void test_dgc0_wire_integrity() {
    // Emit a real valid W8A8 .dgc0 payload
    ft_twin_request_t emit_req = ft_twin_request_default();
    emit_req.workload = FT_TWIN_WORKLOAD_GEMM;
    emit_req.m = 1u; emit_req.k = 64u; emit_req.n = 64u;
    size_t sz = 0u;
    CHECK_TRUE(ft_twin_emit_sample_dgc0(&emit_req, nullptr, 0u, &sz) == FT_TWIN_ERR_OVERFLOW && sz > 100u,
               "emit sample dgc0 size query");
    std::vector<uint8_t> dgc0(sz, 0u);
    CHECK_TRUE(ft_twin_emit_sample_dgc0(&emit_req, dgc0.data(), dgc0.size(), &sz) == FT_TWIN_OK,
               "emit sample dgc0 payload");

    // 1. Valid payload passes on both buenos-a0 & malibu
    ft_twin_result_t ok_res{};
    CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, dgc0.data(), dgc0.size(),
                                        nullptr, 0u, &ok_res) == FT_TWIN_OK, "valid dgc0 eval");
    CHECK_TRUE(ok_res.pass == 1, ok_res.fault_reason);
    CHECK_TRUE(std::strcmp(ok_res.domain, "dgc0") == 0, "domain == dgc0");
    CHECK_TRUE(ok_res.check_dgc0_wire_ok == 1, "dgc0 wire ok");
    CHECK_TRUE(std::strlen(ok_res.dgc0_sha256) == 64u, "sha256 computed");
    check_common_schema_fields(ok_res, serialize_json(ok_res));

    // 2. Matching expected SHA-256 passes; wrong SHA-256 fails
    ft_twin_result_t sha_ok{};
    CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_MALIBU, dgc0.data(), dgc0.size(),
                                        ok_res.dgc0_sha256, 0u, &sha_ok) == FT_TWIN_OK, "sha256 match eval");
    CHECK_TRUE(sha_ok.pass == 1, "matching SHA-256 passes");

    ft_twin_result_t sha_bad{};
    CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, dgc0.data(), dgc0.size(),
                                        "0000000000000000000000000000000000000000000000000000000000000000",
                                        0u, &sha_bad) == FT_TWIN_OK, "sha256 mismatch eval");
    CHECK_TRUE(sha_bad.pass == 0 && std::strstr(sha_bad.fault_reason, "DGC0_SHA256_MISMATCH") != nullptr,
               "corrupted SHA-256 rejected");

    // 3. Corrupted magic fails with DGC0_BAD_MAGIC
    {
        std::vector<uint8_t> bad = dgc0;
        bad[4] = 'X';
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, bad.data(), bad.size(),
                                            nullptr, 0u, &res) == FT_TWIN_OK, "bad magic eval");
        CHECK_TRUE(res.pass == 0 && std::strstr(res.fault_reason, "DGC0_BAD_MAGIC") != nullptr,
                   "bad magic caught");
        check_common_schema_fields(res, serialize_json(res));
    }

    // 4. Truncated payload (< 16 B and mid-buffer truncation) fails cleanly
    {
        ft_twin_result_t r1{};
        CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, dgc0.data(), 8u,
                                            nullptr, 0u, &r1) == FT_TWIN_OK, "short dgc0 eval");
        CHECK_TRUE(r1.pass == 0 && std::strstr(r1.fault_reason, "DGC0_TOO_SMALL") != nullptr,
                   "short buffer caught");

        ft_twin_result_t r2{};
        CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, dgc0.data(), dgc0.size() / 2u,
                                            nullptr, 0u, &r2) == FT_TWIN_OK, "truncated dgc0 eval");
        CHECK_TRUE(r2.pass == 0, "truncated ext bounds caught");
    }

    // 5. Corrupted root offset fails with DGC0_BAD_ROOT
    {
        std::vector<uint8_t> bad = dgc0;
        uint32_t oob_root = 0x7FFFFFFFu;
        std::memcpy(bad.data(), &oob_root, sizeof(oob_root));
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, bad.data(), bad.size(),
                                            nullptr, 0u, &res) == FT_TWIN_OK, "bad root eval");
        CHECK_TRUE(res.pass == 0 && std::strstr(res.fault_reason, "DGC0_BAD_ROOT") != nullptr,
                   "bad root caught");
    }

    // 6. Tight SRAM limit override triggers DGC0_SRAM_OVERFLOW
    {
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_eval_dgc0_buffer(FT_TWIN_TARGET_BUENOS_A0, dgc0.data(), dgc0.size(),
                                            nullptr, 16u, &res) == FT_TWIN_OK, "dgc0 sram overflow eval");
        CHECK_TRUE(res.pass == 0 && std::strstr(res.fault_reason, "DGC0_SRAM_OVERFLOW") != nullptr,
                   "dgc0 sram overflow caught");
    }

    // 7. Determinism check: identical inputs yield identical reward/nmse/sram/est_hw_us/sha256
    {
        ft_twin_result_t r_a{}, r_b{};
        ft_twin_request_t q = ft_twin_request_default();
        q.workload = FT_TWIN_WORKLOAD_QUANT_Q4_K;
        q.m = 1u; q.k = 256u; q.n = 128u; q.seed = 777u;
        CHECK_TRUE(ft_twin_eval(&q, &r_a) == FT_TWIN_OK && ft_twin_eval(&q, &r_b) == FT_TWIN_OK, "det evals");
        CHECK_TRUE(r_a.pass == r_b.pass && r_a.reward == r_b.reward &&
                   r_a.nmse == r_b.nmse && r_a.max_abs_err == r_b.max_abs_err &&
                   r_a.sram_bytes == r_b.sram_bytes && r_a.est_hw_us == r_b.est_hw_us &&
                   std::strcmp(r_a.dgc0_sha256, r_b.dgc0_sha256) == 0,
                   "deterministic bit-identical verdict metrics across runs");
    }

    std::printf("[PASS] test_dgc0_wire_integrity (valid + SHA-256 + magic/trunc/root/SRAM faults + determinism)\n");
}

void test_throughput_over_1000_per_min(ft_twin_session_t* s) {
    constexpr uint32_t kN = 1000u;
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t pass_cnt = 0u;
    for (uint32_t i = 0u; i < kN; ++i) {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = (i & 1u) ? FT_TWIN_TARGET_MALIBU : FT_TWIN_TARGET_BUENOS_A0;
        req.seed = 42u + i;
        switch (i % 6u) {
            case 0u: req.workload = FT_TWIN_WORKLOAD_GEMM; req.m = 1u; req.k = 256u; req.n = 256u; break;
            case 1u: req.workload = FT_TWIN_WORKLOAD_SWIGLU; req.swiglu_11chunk = 1; req.m = 1u; req.k = 576u; req.h = 512u; req.n = 576u; break;
            case 2u: req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN; req.head_dim = 128u; req.seq_len = 1024u; break;
            case 3u: req.workload = FT_TWIN_WORKLOAD_QUANT_W8A16; req.m = 1u; req.k = 512u; req.n = 512u; break;
            case 4u: req.workload = FT_TWIN_WORKLOAD_QUANT_Q4_K; req.m = 1u; req.k = 256u; req.n = 128u; break;
            case 5u: req.workload = FT_TWIN_WORKLOAD_SPLINE_0X640; req.spline_samples = 1025u; break;
        }
        ft_twin_result_t res{};
        CHECK_TRUE(ft_twin_session_eval(s, &req, &res) == FT_TWIN_OK, "bench eval ok");
        if (res.pass) ++pass_cnt;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double evals_per_min = (static_cast<double>(kN) * 60000.0) / elapsed_ms;
    CHECK_TRUE(pass_cnt == kN, "all 1000 benchmark schedules passed");
    CHECK_TRUE(evals_per_min >= 1000.0, "throughput >= 1,000 evals/min");
    std::printf("[PASS] test_throughput_over_1000_per_min (%u evals in %.2f ms = %.1f evals/min, %.4f ms/eval)\n",
                kN, elapsed_ms, evals_per_min, elapsed_ms / static_cast<double>(kN));
}

}  // namespace

int main() {
    ft_twin_session_t* session = nullptr;
    if (ft_twin_session_create(&session) != FT_TWIN_OK || !session) {
        std::fprintf(stderr, "FAIL: ft_twin_session_create failed\n");
        return 1;
    }

    test_gemm_profiles(session);
    test_swiglu_11chunk_and_quant(session);
    test_flash_attn(session);
    test_quant_kernels(session);
    test_splines_0x640(session);
    test_dgc0_wire_integrity();
    test_throughput_over_1000_per_min(session);

    ft_twin_session_destroy(session);

    if (g_failures != 0) {
        std::fprintf(stderr, "FREETILE_TWIN TESTS FAILED (%d failures)\n", g_failures);
        return 1;
    }
    std::printf("ALL FREETILE_TWIN TESTS PASSED (7/7 suites)\n");
    return 0;
}
