// SPDX-License-Identifier: Apache-2.0
// freetile-twin — Fast deterministic CLI for FreeTile Digital Twin evaluation,
// batch JSONL RL/swarm scoring, .dgc0 wire verification, and throughput benchmarking.

#include "freetile/twin.h"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using SteadyClock = std::chrono::steady_clock;

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool parse_target(const std::string& s, ft_twin_target_t* out) {
    const std::string l = to_lower(s);
    if (l == "buenos-a0" || l == "buenos_a0" || l == "buenos") {
        *out = FT_TWIN_TARGET_BUENOS_A0;
        return true;
    }
    if (l == "malibu" || l == "malibu-a0" || l == "malibu_a0") {
        *out = FT_TWIN_TARGET_MALIBU;
        return true;
    }
    return false;
}

bool parse_workload(const std::string& s, ft_twin_workload_t* out) {
    const std::string l = to_lower(s);
    if (l == "gemm") { *out = FT_TWIN_WORKLOAD_GEMM; return true; }
    if (l == "swiglu") { *out = FT_TWIN_WORKLOAD_SWIGLU; return true; }
    if (l == "flash_attn" || l == "flash-attn" || l == "flashattn") { *out = FT_TWIN_WORKLOAD_FLASH_ATTN; return true; }
    if (l == "w8a16") { *out = FT_TWIN_WORKLOAD_QUANT_W8A16; return true; }
    if (l == "q4_k" || l == "q4k") { *out = FT_TWIN_WORKLOAD_QUANT_Q4_K; return true; }
    if (l == "q6_k" || l == "q6k") { *out = FT_TWIN_WORKLOAD_QUANT_Q6_K; return true; }
    if (l == "iq4_nl" || l == "iq4nl") { *out = FT_TWIN_WORKLOAD_QUANT_IQ4_NL; return true; }
    if (l == "iq4_xs" || l == "iq4xs") { *out = FT_TWIN_WORKLOAD_QUANT_IQ4_XS; return true; }
    if (l == "q8_0" || l == "q80") { *out = FT_TWIN_WORKLOAD_QUANT_Q8_0; return true; }
    if (l == "q4_0" || l == "q40") { *out = FT_TWIN_WORKLOAD_QUANT_Q4_0; return true; }
    if (l == "spline" || l == "0x640" || l == "spline_0x640") { *out = FT_TWIN_WORKLOAD_SPLINE_0X640; return true; }
    if (l == "dgc0") { *out = FT_TWIN_WORKLOAD_DGC0; return true; }
    return false;
}

bool parse_dtype(const std::string& s, ft_twin_dtype_t* out) {
    const std::string l = to_lower(s);
    if (l == "bf16") { *out = FT_TWIN_DTYPE_BF16; return true; }
    if (l == "fp32" || l == "f32") { *out = FT_TWIN_DTYPE_FP32; return true; }
    if (l == "fp16" || l == "f16") { *out = FT_TWIN_DTYPE_FP16; return true; }
    if (l == "int8" || l == "s8" || l == "i8" || l == "w8a8") { *out = FT_TWIN_DTYPE_INT8; return true; }
    return false;
}

bool parse_activation(const std::string& s, ft_twin_activation_t* out) {
    const std::string l = to_lower(s);
    if (l == "none") { *out = FT_TWIN_ACT_NONE; return true; }
    if (l == "relu") { *out = FT_TWIN_ACT_RELU; return true; }
    if (l == "relu6") { *out = FT_TWIN_ACT_RELU6; return true; }
    if (l == "silu") { *out = FT_TWIN_ACT_SILU; return true; }
    if (l == "gelu") { *out = FT_TWIN_ACT_GELU; return true; }
    return false;
}

bool parse_spline_kind(const std::string& s, ft_twin_spline_kind_t* out) {
    const std::string l = to_lower(s);
    if (l == "exp" || l == "exp_neg12_96") { *out = FT_TWIN_SPLINE_EXP_NEG12_96; return true; }
    if (l == "exp_neg11") { *out = FT_TWIN_SPLINE_EXP_NEG11; return true; }
    if (l == "rsqrt") { *out = FT_TWIN_SPLINE_RSQRT; return true; }
    if (l == "recip") { *out = FT_TWIN_SPLINE_RECIP; return true; }
    if (l == "silu") { *out = FT_TWIN_SPLINE_SILU; return true; }
    if (l == "gelu" || l == "gelu_tanh") { *out = FT_TWIN_SPLINE_GELU_TANH; return true; }
    if (l == "gelu_erf") { *out = FT_TWIN_SPLINE_GELU_ERF; return true; }
    return false;
}

bool parse_quant_type_u32(const std::string& s, uint32_t* out) {
    const std::string l = to_lower(s);
    if (l == "q8_0") { *out = 0u; return true; }
    if (l == "q4_0") { *out = 1u; return true; }
    if (l == "q4_k") { *out = 2u; return true; }
    if (l == "q6_k") { *out = 3u; return true; }
    if (l == "iq4_nl") { *out = 4u; return true; }
    if (l == "iq4_xs") { *out = 5u; return true; }
    return false;
}

bool parse_fa_phase(const std::string& s, ft_twin_fa_phase_t* out) {
    const std::string l = to_lower(s);
    if (l == "decode") { *out = FT_TWIN_FA_DECODE; return true; }
    if (l == "prefill") { *out = FT_TWIN_FA_PREFILL; return true; }
    return false;
}

bool parse_fa_policy(const std::string& s, ft_twin_fa_buf_policy_t* out) {
    const std::string l = to_lower(s);
    if (l == "ping_pong" || l == "ping_pong_full") { *out = FT_TWIN_FA_PING_PONG; return true; }
    if (l == "kv_db_fp32" || l == "kv_double_buffered_fp32_acc") { *out = FT_TWIN_FA_KV_DB_FP32; return true; }
    if (l == "single" || l == "single_buffered") { *out = FT_TWIN_FA_SINGLE_BUF; return true; }
    return false;
}

bool parse_kv_format(const std::string& s, ft_twin_kv_format_t* out) {
    const std::string l = to_lower(s);
    if (l == "q8_0") { *out = FT_TWIN_KV_Q8_0; return true; }
    if (l == "f16" || l == "fp16") { *out = FT_TWIN_KV_F16; return true; }
    if (l == "q4_0") { *out = FT_TWIN_KV_Q4_0; return true; }
    return false;
}

bool print_result_json(const ft_twin_result_t& res, int pretty) {
    size_t need = 0u;
    ft_twin_status_t st = ft_twin_result_to_json(&res, pretty, nullptr, 0u, &need);
    if (st != FT_TWIN_ERR_OVERFLOW && st != FT_TWIN_OK) return false;
    std::vector<char> buf(need + 1u, '\0');
    st = ft_twin_result_to_json(&res, pretty, buf.data(), buf.size(), &need);
    if (st != FT_TWIN_OK) return false;
    std::fwrite(buf.data(), 1u, need, stdout);
    std::fputc('\n', stdout);
    return true;
}

// Zero-dependency flat JSON object parser for --batch-jsonl stdin mode.
struct ParsedLineStrings {
    std::string workload_label;
    std::string dgc0_path;
    std::string expected_sha256;
};

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

bool parse_json_string(const std::string& s, size_t& i, std::string* out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    out->clear();
    while (i < s.size()) {
        const char c = s[i++];
        if (c == '"') return true;
        if (c == '\\' && i < s.size()) {
            const char e = s[i++];
            switch (e) {
                case '"':  out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/':  out->push_back('/'); break;
                case 'n':  out->push_back('\n'); break;
                case 'r':  out->push_back('\r'); break;
                case 't':  out->push_back('\t'); break;
                default:   out->push_back(e); break;
            }
        } else {
            out->push_back(c);
        }
    }
    return false;
}

bool parse_json_token(const std::string& s, size_t& i, std::string* out, bool* is_str) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    if (s[i] == '"') {
        *is_str = true;
        return parse_json_string(s, i, out);
    }
    *is_str = false;
    const size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && !std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    if (i == start) return false;
    *out = s.substr(start, i - start);
    return true;
}

bool parse_jsonl_request(const std::string& line,
                         ft_twin_request_t* out_req,
                         ParsedLineStrings* store,
                         std::string* err) {
    *out_req = ft_twin_request_default();
    *store = ParsedLineStrings{};

    size_t i = 0u;
    skip_ws(line, i);
    if (i >= line.size() || line[i] != '{') {
        if (err) *err = "Expected JSON object starting with '{'";
        return false;
    }
    ++i;
    while (i < line.size()) {
        skip_ws(line, i);
        if (i < line.size() && line[i] == '}') {
            ++i;
            break;
        }
        std::string key;
        if (!parse_json_string(line, i, &key)) {
            if (err) *err = "Invalid JSON key";
            return false;
        }
        skip_ws(line, i);
        if (i >= line.size() || line[i] != ':') {
            if (err) *err = "Expected ':' after JSON key";
            return false;
        }
        ++i;
        std::string val;
        bool is_str = false;
        if (!parse_json_token(line, i, &val, &is_str)) {
            if (err) *err = "Invalid JSON value for key " + key;
            return false;
        }

        const auto to_u32 = [&]() -> uint32_t { return static_cast<uint32_t>(std::strtoul(val.c_str(), nullptr, 10)); };
        const auto to_u64 = [&]() -> uint64_t { return static_cast<uint64_t>(std::strtoull(val.c_str(), nullptr, 10)); };
        const auto to_i32 = [&]() -> int {
            if (val == "true") return 1;
            if (val == "false") return 0;
            return static_cast<int>(std::strtol(val.c_str(), nullptr, 10));
        };
        const auto to_f64 = [&]() -> double { return std::strtod(val.c_str(), nullptr); };

        if (key == "target") {
            if (!parse_target(val, &out_req->target)) { if (err) *err = "Unknown target: " + val; return false; }
        } else if (key == "workload") {
            if (!parse_workload(val, &out_req->workload)) { if (err) *err = "Unknown workload: " + val; return false; }
        } else if (key == "workload_label") {
            store->workload_label = val;
            out_req->workload_label = store->workload_label.c_str();
        } else if (key == "m") { out_req->m = to_u32(); }
        else if (key == "k") { out_req->k = to_u32(); }
        else if (key == "n") { out_req->n = to_u32(); }
        else if (key == "h") { out_req->h = to_u32(); }
        else if (key == "bm") { out_req->bm = to_u32(); }
        else if (key == "bn") { out_req->bn = to_u32(); }
        else if (key == "bk") { out_req->bk = to_u32(); }
        else if (key == "double_buffered") { out_req->double_buffered = to_i32(); }
        else if (key == "act_dtype") { parse_dtype(val, &out_req->act_dtype); }
        else if (key == "out_dtype") { parse_dtype(val, &out_req->out_dtype); }
        else if (key == "activation") { parse_activation(val, &out_req->activation); }
        else if (key == "swiglu_11chunk") { out_req->swiglu_11chunk = to_i32(); }
        else if (key == "quant" || key == "quant_type") {
            if (is_str) parse_quant_type_u32(val, &out_req->quant_type);
            else out_req->quant_type = to_u32();
        }
        else if (key == "fa_phase") { parse_fa_phase(val, &out_req->fa_phase); }
        else if (key == "head_dim") { out_req->head_dim = to_u32(); }
        else if (key == "num_q_heads") { out_req->num_q_heads = to_u32(); }
        else if (key == "num_kv_heads") { out_req->num_kv_heads = to_u32(); }
        else if (key == "seq_len") { out_req->seq_len = to_u32(); }
        else if (key == "br") { out_req->br = to_u32(); }
        else if (key == "bc") { out_req->bc = to_u32(); }
        else if (key == "kv_tile_align") { out_req->kv_tile_align = to_u32(); }
        else if (key == "fa_policy" || key == "fa_buf_policy") { parse_fa_policy(val, &out_req->fa_buf_policy); }
        else if (key == "kv_format") { parse_kv_format(val, &out_req->kv_format); }
        else if (key == "spline" || key == "spline_kind") { parse_spline_kind(val, &out_req->spline_kind); }
        else if (key == "spline_samples") { out_req->spline_samples = to_u32(); }
        else if (key == "sram_limit" || key == "sram_limit_override_bytes") { out_req->sram_limit_override_bytes = to_u64(); }
        else if (key == "max_nmse") { out_req->max_nmse = to_f64(); }
        else if (key == "max_abs_err_limit") { out_req->max_abs_err_limit = to_f64(); }
        else if (key == "malibu_threads") { out_req->malibu_threads = to_u32(); }
        else if (key == "malibu_active_engine_mask") { out_req->malibu_active_engine_mask = to_u32(); }
        else if (key == "malibu_mesa_pco_ready") { out_req->malibu_mesa_pco_ready = to_i32(); }
        else if (key == "seed") { out_req->seed = to_u32(); }
        else if (key == "verify_dgc0") { out_req->verify_dgc0 = to_i32(); }
        else if (key == "dgc0_path" || key == "dgc0") {
            store->dgc0_path = val;
            out_req->dgc0_path = store->dgc0_path.c_str();
            out_req->workload = FT_TWIN_WORKLOAD_DGC0;
        } else if (key == "expected_sha256") {
            store->expected_sha256 = val;
            out_req->expected_sha256 = store->expected_sha256.c_str();
        }

        skip_ws(line, i);
        if (i < line.size() && line[i] == ',') {
            ++i;
            continue;
        }
        if (i < line.size() && line[i] == '}') {
            ++i;
            break;
        }
    }
    return true;
}

// Build a diverse deterministic corpus of candidate schedules covering ALL workloads
// (GEMM BF16/FP32/W8A8, 11-chunk & quant SwiGLU, FlashAttention, W8A16, Q4_K, Q6_K,
// IQ4_NL, IQ4_XS, 0x640 splines, and .dgc0 payloads) across both buenos-a0 and malibu.
int run_benchmark(uint32_t count, bool emit_jsonl) {
    ft_twin_session_t* session = nullptr;
    if (ft_twin_session_create(&session) != FT_TWIN_OK) {
        std::fprintf(stderr, "ERROR: failed to create freetile-twin session\n");
        return 1;
    }

    // Generate valid & corrupted in-memory .dgc0 payloads for the benchmark mix.
    ft_twin_request_t sample_req = ft_twin_request_default();
    sample_req.workload = FT_TWIN_WORKLOAD_GEMM;
    sample_req.m = 1u; sample_req.k = 64u; sample_req.n = 64u;
    size_t dgc_sz = 0u;
    (void)ft_twin_emit_sample_dgc0(&sample_req, nullptr, 0u, &dgc_sz);
    std::vector<uint8_t> valid_dgc0(dgc_sz, 0u);
    (void)ft_twin_emit_sample_dgc0(&sample_req, valid_dgc0.data(), valid_dgc0.size(), &dgc_sz);
    std::vector<uint8_t> bad_dgc0 = valid_dgc0;
    if (bad_dgc0.size() > 7u) bad_dgc0[4] = 'X';

    uint32_t passed = 0u;
    uint32_t failed = 0u;
    uint64_t json_bytes_total = 0u;
    std::vector<char> jbuf(4096u, '\0');

    const auto t0 = SteadyClock::now();
    for (uint32_t i = 0u; i < count; ++i) {
        ft_twin_request_t req = ft_twin_request_default();
        req.target = ((i & 1u) == 0u) ? FT_TWIN_TARGET_BUENOS_A0 : FT_TWIN_TARGET_MALIBU;
        req.seed = 1000u + i;
        req.verify_dgc0 = 1;

        const uint32_t kind = i % 12u;
        switch (kind) {
            case 0u: // GEMM BF16 + SiLU
                req.workload = FT_TWIN_WORKLOAD_GEMM;
                req.m = 1u; req.k = 256u; req.n = 256u;
                req.act_dtype = FT_TWIN_DTYPE_BF16;
                req.out_dtype = FT_TWIN_DTYPE_FP32;
                req.activation = FT_TWIN_ACT_SILU;
                break;
            case 1u: // GEMM FP32 + GELU
                req.workload = FT_TWIN_WORKLOAD_GEMM;
                req.m = 1u; req.k = 128u; req.n = 128u;
                req.act_dtype = FT_TWIN_DTYPE_FP32;
                req.out_dtype = FT_TWIN_DTYPE_FP32;
                req.activation = FT_TWIN_ACT_GELU;
                break;
            case 2u: // 11-chunk SwiGLU
                req.workload = FT_TWIN_WORKLOAD_SWIGLU;
                req.swiglu_11chunk = 1;
                req.m = 1u; req.k = 576u; req.h = 512u; req.n = 576u;
                break;
            case 3u: // FlashAttention GQA decode
                req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
                req.fa_phase = FT_TWIN_FA_DECODE;
                req.head_dim = 128u; req.num_q_heads = 32u; req.num_kv_heads = 4u;
                req.seq_len = 1024u; req.kv_format = FT_TWIN_KV_Q8_0;
                break;
            case 4u: // W8A16 1x512x512
                req.workload = FT_TWIN_WORKLOAD_QUANT_W8A16;
                req.m = 1u; req.k = 512u; req.n = 512u;
                break;
            case 5u: // Q4_K
                req.workload = FT_TWIN_WORKLOAD_QUANT_Q4_K;
                req.m = 1u; req.k = 256u; req.n = 128u;
                break;
            case 6u: // Q6_K
                req.workload = FT_TWIN_WORKLOAD_QUANT_Q6_K;
                req.m = 1u; req.k = 256u; req.n = 128u;
                break;
            case 7u: // IQ4_NL
                req.workload = FT_TWIN_WORKLOAD_QUANT_IQ4_NL;
                req.m = 1u; req.k = 256u; req.n = 128u;
                break;
            case 8u: // IQ4_XS
                req.workload = FT_TWIN_WORKLOAD_QUANT_IQ4_XS;
                req.m = 1u; req.k = 256u; req.n = 128u;
                break;
            case 9u: // 0x640 Spline
                req.workload = FT_TWIN_WORKLOAD_SPLINE_0X640;
                req.spline_kind = static_cast<ft_twin_spline_kind_t>((i / 12u) % 7u);
                req.spline_samples = 1025u;
                break;
            case 10u: // Valid .dgc0 payload
                req.workload = FT_TWIN_WORKLOAD_DGC0;
                req.dgc0_bytes = valid_dgc0.data();
                req.dgc0_size = valid_dgc0.size();
                break;
            case 11u: // Negative fault mix (alternating SRAM overflow & corrupted .dgc0)
                if (((i / 12u) & 1u) == 0u) {
                    req.workload = FT_TWIN_WORKLOAD_FLASH_ATTN;
                    req.fa_phase = FT_TWIN_FA_PREFILL;
                    req.head_dim = 128u; req.br = 64u; req.bc = 1024u; // Exceeds 768 KiB narrow SRAM
                } else {
                    req.workload = FT_TWIN_WORKLOAD_DGC0;
                    req.dgc0_bytes = bad_dgc0.data();
                    req.dgc0_size = bad_dgc0.size();
                }
                break;
        }

        ft_twin_result_t res{};
        if (ft_twin_session_eval(session, &req, &res) != FT_TWIN_OK) {
            std::fprintf(stderr, "ERROR: evaluation %u failed internally\n", i);
            ft_twin_session_destroy(session);
            return 1;
        }
        if (res.pass) ++passed; else ++failed;

        size_t jlen = 0u;
        ft_twin_status_t jst = ft_twin_result_to_json(&res, 0, jbuf.data(), jbuf.size(), &jlen);
        if (jst == FT_TWIN_ERR_OVERFLOW) {
            jbuf.resize(jlen + 1u);
            jst = ft_twin_result_to_json(&res, 0, jbuf.data(), jbuf.size(), &jlen);
        }
        if (jst == FT_TWIN_OK) {
            json_bytes_total += jlen;
            if (emit_jsonl) {
                std::fwrite(jbuf.data(), 1u, jlen, stdout);
                std::fputc('\n', stdout);
            }
        }
    }
    const auto t1 = SteadyClock::now();
    ft_twin_session_destroy(session);

    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per_eval = (count > 0u) ? (elapsed_ms / static_cast<double>(count)) : 0.0;
    const double evals_per_sec = (elapsed_ms > 0.0) ? (static_cast<double>(count) * 1000.0 / elapsed_ms) : 0.0;
    const double evals_per_min = evals_per_sec * 60.0;

    if (!emit_jsonl) {
        std::printf("{\"schema_version\":\"%s\",\"mode\":\"bench\",\"total_evals\":%u,"
                    "\"passed\":%u,\"failed\":%u,\"elapsed_ms\":%.3f,\"ms_per_eval\":%.4f,"
                    "\"evals_per_sec\":%.2f,\"evals_per_min\":%.2f,\"json_bytes\":%llu,"
                    "\"meets_1000_per_min\":%s}\n",
                    FT_TWIN_SCHEMA_VERSION,
                    count,
                    passed,
                    failed,
                    elapsed_ms,
                    ms_per_eval,
                    evals_per_sec,
                    evals_per_min,
                    static_cast<unsigned long long>(json_bytes_total),
                    (evals_per_min >= 1000.0) ? "true" : "false");
    }
    return 0;
}

void print_usage() {
    std::printf(
        "freetile-twin — FreeTPU / FreeTile Digital Twin (schema: %s)\n"
        "\n"
        "Usage:\n"
        "  freetile-twin [options]\n"
        "  freetile-twin --dgc0 <payload.dgc0> [--target buenos-a0|malibu] [--expected-sha256 <hex64>]\n"
        "  freetile-twin --batch-jsonl < candidates.jsonl\n"
        "  freetile-twin --bench [N] [--bench-jsonl]\n"
        "\n"
        "Options:\n"
        "  --target <buenos-a0|malibu>           Hardware target profile (default: buenos-a0)\n"
        "  --workload <kind>                     gemm | swiglu | flash_attn | w8a16 | q4_k | q6_k |\n"
        "                                        iq4_nl | iq4_xs | q8_0 | q4_0 | spline | dgc0\n"
        "  --m <u32> --k <u32> --n <u32> --h <u32>   Matrix / SwiGLU dimensions\n"
        "  --bm <u32> --bn <u32> --bk <u32>      Candidate tile overrides (0 = auto)\n"
        "  --double-buffered <0|1|auto>          Double-buffering override\n"
        "  --act-dtype <bf16|fp32|fp16|int8>     Activation dtype\n"
        "  --out-dtype <bf16|fp32|fp16|int8>     Output dtype\n"
        "  --activation <none|relu|relu6|silu|gelu>  Fused epilogue activation\n"
        "  --swiglu-11chunk <0|1>                11-chunk chained INT8 SwiGLU vs quantized SwiGLU\n"
        "  --quant <q8_0|q4_0|q4_k|q6_k|iq4_nl|iq4_xs>  Quant format for quantized SwiGLU\n"
        "  --fa-phase <decode|prefill>           FlashAttention phase\n"
        "  --head-dim <u32> --q-heads <u32> --kv-heads <u32> --seq-len <u32>\n"
        "  --br <u32> --bc <u32> --kv-align <u32>    FlashAttention tile overrides\n"
        "  --fa-policy <ping_pong|kv_db_fp32|single> FlashAttention SRAM buffer policy\n"
        "  --kv-format <q8_0|f16|q4_0>           KV cache DMA-BUF format\n"
        "  --spline <exp|exp_neg11|rsqrt|recip|silu|gelu|gelu_erf>  0x640 spline kind\n"
        "  --spline-samples <u32>                Spline verification sample count\n"
        "  --dgc0 <path>                         Validate raw .dgc0 binary file\n"
        "  --expected-sha256 <hex64>             Verify .dgc0 SHA-256 digest\n"
        "  --emit-dgc0 <path>                    Emit deterministic .dgc0 payload to file\n"
        "  --sram-limit <bytes>                  Override target SRAM limit in bytes\n"
        "  --max-nmse <f64>                      Override max allowed NMSE\n"
        "  --no-dgc0                             Skip .dgc0 emission check on schedule evals\n"
        "  --seed <u32>                          Deterministic PRNG seed (default: 42)\n"
        "  --batch-jsonl                         Read JSONL candidate requests from stdin\n"
        "  --bench [N]                           Run N diverse real evaluations (default: 1000)\n"
        "  --bench-jsonl                         Emit each benchmark verdict as JSONL\n"
        "  --pretty                              Pretty-print JSON output\n"
        "  --help                                Show this help message\n",
        FT_TWIN_SCHEMA_VERSION);
}

}  // namespace

int main(int argc, char** argv) {
    ft_twin_request_t req = ft_twin_request_default();
    std::string dgc0_path_str;
    std::string expected_sha_str;
    std::string emit_dgc0_path;
    std::string workload_label_str;

    bool batch_jsonl = false;
    bool bench_mode = false;
    bool bench_jsonl = false;
    uint32_t bench_count = 1000u;
    int pretty = 0;

    auto need_arg = [&](int& idx, const char* flag) -> const char* {
        if (idx + 1 >= argc) {
            std::fprintf(stderr, "ERROR: missing value for %s\n", flag);
            std::exit(2);
        }
        return argv[++idx];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--pretty") {
            pretty = 1;
        } else if (arg == "--batch-jsonl") {
            batch_jsonl = true;
        } else if (arg == "--bench-jsonl") {
            bench_jsonl = true;
        } else if (arg == "--bench") {
            bench_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bench_count = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
                if (bench_count == 0u) bench_count = 1000u;
            }
        } else if (arg == "--target") {
            if (!parse_target(need_arg(i, "--target"), &req.target)) {
                std::fprintf(stderr, "ERROR: invalid --target (expected buenos-a0|malibu)\n");
                return 2;
            }
        } else if (arg == "--workload") {
            if (!parse_workload(need_arg(i, "--workload"), &req.workload)) {
                std::fprintf(stderr, "ERROR: invalid --workload\n");
                return 2;
            }
        } else if (arg == "--workload-label") {
            workload_label_str = need_arg(i, "--workload-label");
            req.workload_label = workload_label_str.c_str();
        } else if (arg == "--m") { req.m = static_cast<uint32_t>(std::strtoul(need_arg(i, "--m"), nullptr, 10)); }
        else if (arg == "--k") { req.k = static_cast<uint32_t>(std::strtoul(need_arg(i, "--k"), nullptr, 10)); }
        else if (arg == "--n") { req.n = static_cast<uint32_t>(std::strtoul(need_arg(i, "--n"), nullptr, 10)); }
        else if (arg == "--h") { req.h = static_cast<uint32_t>(std::strtoul(need_arg(i, "--h"), nullptr, 10)); }
        else if (arg == "--bm") { req.bm = static_cast<uint32_t>(std::strtoul(need_arg(i, "--bm"), nullptr, 10)); }
        else if (arg == "--bn") { req.bn = static_cast<uint32_t>(std::strtoul(need_arg(i, "--bn"), nullptr, 10)); }
        else if (arg == "--bk") { req.bk = static_cast<uint32_t>(std::strtoul(need_arg(i, "--bk"), nullptr, 10)); }
        else if (arg == "--double-buffered") {
            const std::string v = to_lower(need_arg(i, "--double-buffered"));
            if (v == "auto") req.double_buffered = -1;
            else req.double_buffered = (v == "1" || v == "true") ? 1 : 0;
        } else if (arg == "--act-dtype") {
            if (!parse_dtype(need_arg(i, "--act-dtype"), &req.act_dtype)) {
                std::fprintf(stderr, "ERROR: invalid --act-dtype\n");
                return 2;
            }
        } else if (arg == "--out-dtype") {
            if (!parse_dtype(need_arg(i, "--out-dtype"), &req.out_dtype)) {
                std::fprintf(stderr, "ERROR: invalid --out-dtype\n");
                return 2;
            }
        } else if (arg == "--activation") {
            if (!parse_activation(need_arg(i, "--activation"), &req.activation)) {
                std::fprintf(stderr, "ERROR: invalid --activation\n");
                return 2;
            }
        } else if (arg == "--swiglu-11chunk") {
            req.swiglu_11chunk = (std::strtol(need_arg(i, "--swiglu-11chunk"), nullptr, 10) != 0) ? 1 : 0;
        } else if (arg == "--quant") {
            if (!parse_quant_type_u32(need_arg(i, "--quant"), &req.quant_type)) {
                std::fprintf(stderr, "ERROR: invalid --quant\n");
                return 2;
            }
        } else if (arg == "--fa-phase") {
            if (!parse_fa_phase(need_arg(i, "--fa-phase"), &req.fa_phase)) {
                std::fprintf(stderr, "ERROR: invalid --fa-phase\n");
                return 2;
            }
        } else if (arg == "--head-dim") { req.head_dim = static_cast<uint32_t>(std::strtoul(need_arg(i, "--head-dim"), nullptr, 10)); }
        else if (arg == "--q-heads") { req.num_q_heads = static_cast<uint32_t>(std::strtoul(need_arg(i, "--q-heads"), nullptr, 10)); }
        else if (arg == "--kv-heads") { req.num_kv_heads = static_cast<uint32_t>(std::strtoul(need_arg(i, "--kv-heads"), nullptr, 10)); }
        else if (arg == "--seq-len") { req.seq_len = static_cast<uint32_t>(std::strtoul(need_arg(i, "--seq-len"), nullptr, 10)); }
        else if (arg == "--br") { req.br = static_cast<uint32_t>(std::strtoul(need_arg(i, "--br"), nullptr, 10)); }
        else if (arg == "--bc") { req.bc = static_cast<uint32_t>(std::strtoul(need_arg(i, "--bc"), nullptr, 10)); }
        else if (arg == "--kv-align") { req.kv_tile_align = static_cast<uint32_t>(std::strtoul(need_arg(i, "--kv-align"), nullptr, 10)); }
        else if (arg == "--fa-policy") {
            if (!parse_fa_policy(need_arg(i, "--fa-policy"), &req.fa_buf_policy)) {
                std::fprintf(stderr, "ERROR: invalid --fa-policy\n");
                return 2;
            }
        } else if (arg == "--kv-format") {
            if (!parse_kv_format(need_arg(i, "--kv-format"), &req.kv_format)) {
                std::fprintf(stderr, "ERROR: invalid --kv-format\n");
                return 2;
            }
        } else if (arg == "--spline") {
            if (!parse_spline_kind(need_arg(i, "--spline"), &req.spline_kind)) {
                std::fprintf(stderr, "ERROR: invalid --spline\n");
                return 2;
            }
        } else if (arg == "--spline-samples") {
            req.spline_samples = static_cast<uint32_t>(std::strtoul(need_arg(i, "--spline-samples"), nullptr, 10));
        } else if (arg == "--dgc0") {
            dgc0_path_str = need_arg(i, "--dgc0");
            req.dgc0_path = dgc0_path_str.c_str();
            req.workload = FT_TWIN_WORKLOAD_DGC0;
        } else if (arg == "--expected-sha256") {
            expected_sha_str = need_arg(i, "--expected-sha256");
            req.expected_sha256 = expected_sha_str.c_str();
        } else if (arg == "--emit-dgc0") {
            emit_dgc0_path = need_arg(i, "--emit-dgc0");
        } else if (arg == "--sram-limit") {
            req.sram_limit_override_bytes = static_cast<uint64_t>(std::strtoull(need_arg(i, "--sram-limit"), nullptr, 10));
        } else if (arg == "--max-nmse") {
            req.max_nmse = std::strtod(need_arg(i, "--max-nmse"), nullptr);
        } else if (arg == "--no-dgc0") {
            req.verify_dgc0 = 0;
        } else if (arg == "--seed") {
            req.seed = static_cast<uint32_t>(std::strtoul(need_arg(i, "--seed"), nullptr, 10));
        } else {
            std::fprintf(stderr, "ERROR: unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (bench_mode) {
        return run_benchmark(bench_count, bench_jsonl);
    }

    if (!emit_dgc0_path.empty()) {
        size_t sz = 0u;
        ft_twin_status_t st = ft_twin_emit_sample_dgc0(&req, nullptr, 0u, &sz);
        if (st != FT_TWIN_ERR_OVERFLOW && st != FT_TWIN_OK) {
            std::fprintf(stderr, "ERROR: failed to emit .dgc0 for requested workload\n");
            return 1;
        }
        std::vector<uint8_t> buf(sz, 0u);
        st = ft_twin_emit_sample_dgc0(&req, buf.data(), buf.size(), &sz);
        if (st != FT_TWIN_OK) {
            std::fprintf(stderr, "ERROR: failed to serialize .dgc0\n");
            return 1;
        }
        std::ofstream ofs(emit_dgc0_path, std::ios::binary);
        if (!ofs || !ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()))) {
            std::fprintf(stderr, "ERROR: failed to write %s\n", emit_dgc0_path.c_str());
            return 1;
        }
        std::printf("{\"emitted_dgc0\":\"%s\",\"bytes\":%zu}\n", emit_dgc0_path.c_str(), buf.size());
        return 0;
    }

    if (batch_jsonl) {
        ft_twin_session_t* session = nullptr;
        if (ft_twin_session_create(&session) != FT_TWIN_OK) {
            std::fprintf(stderr, "ERROR: failed to create freetile-twin session\n");
            return 1;
        }
        std::string line;
        while (std::getline(std::cin, line)) {
            size_t p = 0u;
            skip_ws(line, p);
            if (p >= line.size()) continue;

            ft_twin_request_t line_req{};
            ParsedLineStrings store;
            std::string perr;
            if (!parse_jsonl_request(line, &line_req, &store, &perr)) {
                ft_twin_result_t err_res{};
                std::snprintf(err_res.schema_version, sizeof(err_res.schema_version), "%s", FT_TWIN_SCHEMA_VERSION);
                std::snprintf(err_res.domain, sizeof(err_res.domain), "freetile");
                std::snprintf(err_res.target, sizeof(err_res.target), "buenos-a0");
                std::snprintf(err_res.workload, sizeof(err_res.workload), "invalid_jsonl");
                err_res.pass = 0;
                err_res.reward = -1.0;
                err_res.has_fault = 1;
                std::snprintf(err_res.fault_reason, sizeof(err_res.fault_reason), "JSONL_PARSE_ERROR: %s", perr.c_str());
                err_res.check_sram_budget_ok = -1;
                err_res.check_dgc0_wire_ok = -1;
                err_res.check_spline_0x640_ok = -1;
                err_res.check_reference_parity_ok = -1;
                err_res.check_profile_routing_ok = -1;
                err_res.check_schedule_valid = 0;
                print_result_json(err_res, 0);
                continue;
            }
            ft_twin_result_t res{};
            ft_twin_session_eval(session, &line_req, &res);
            print_result_json(res, pretty);
        }
        ft_twin_session_destroy(session);
        return 0;
    }

    ft_twin_result_t res{};
    if (ft_twin_eval(&req, &res) != FT_TWIN_OK) {
        std::fprintf(stderr, "ERROR: ft_twin_eval failed\n");
        return 1;
    }
    if (!print_result_json(res, pretty)) {
        std::fprintf(stderr, "ERROR: JSON serialization failed\n");
        return 1;
    }
    return res.pass ? 0 : 1;
}
