// Clean-room zero-Google-.so FreeTPU Tile DSL runner for Edge TPU (buenos_a0 / rango).
// Links statically against libopenbuenos.a and drives /dev/edgetpu-soc directly
// (or host mock mode with --mock / OPENBUENOS_MOCK=1).

#include "openbuenos.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string json_str(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 0x20u) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
    else out << static_cast<char>(c);
  }
  out << '"';
  return out.str();
}

struct RunnerArgs {
  std::string op = "matmul_silu";
  std::string dtype = "w8a16";
  uint32_t m = 64u;
  uint32_t k = 512u;
  uint32_t n = 512u;
  uint32_t n_inter = 0u;
  std::string in_path;
  std::string w_path;
  std::string w_up_path;
  std::string w_down_path;
  std::string bias_path;
  std::string out_path;
  std::string dive_interp_path = "/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin";
  float s_in = 0.03125f;
  int32_t zp_in = 0;
  float s_w = 0.015625f;
  float s_out = 0.0625f;
  int32_t zp_out = 0;
  int warmup = 3;
  int iters = 20;
  int perf_mode = 5;
  bool mock = false;
};

bool read_bin(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    *err = "Cannot open input file: " + path;
    return false;
  }
  std::streamsize sz = ifs.tellg();
  if (sz < 0) {
    *err = "Failed to get size of: " + path;
    return false;
  }
  ifs.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(sz));
  if (sz > 0 && !ifs.read(reinterpret_cast<char*>(out->data()), sz)) {
    *err = "Failed reading: " + path;
    return false;
  }
  return true;
}

bool write_bin(const std::string& path, const void* data, size_t bytes, std::string* err) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    *err = "Cannot open output file: " + path;
    return false;
  }
  if (bytes > 0 && !ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes))) {
    *err = "Failed writing: " + path;
    return false;
  }
  return true;
}

uint16_t fp32_to_fp16(float f) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = bits & 0x7FFFFFu;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float fp16_to_fp32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t bits = 0u;
  if (exp == 0u) {
    bits = sign;
  } else if (exp == 31u) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

void quantize_f32_to_q8_0(const float* src_nk, uint32_t n_dim, uint32_t k_dim,
                          uint32_t k_pad, std::vector<ob_block_q8_0_t>* out) {
  const uint32_t nb_k = k_pad / OB_QK8_0;
  out->assign(static_cast<size_t>(n_dim) * nb_k, ob_block_q8_0_t{});
  for (uint32_t n = 0; n < n_dim; ++n) {
    const float* row = src_nk + static_cast<size_t>(n) * k_dim;
    for (uint32_t bi = 0; bi < nb_k; ++bi) {
      float amax = 0.0f;
      for (uint32_t j = 0; j < OB_QK8_0; ++j) {
        uint32_t k = bi * OB_QK8_0 + j;
        float v = (k < k_dim) ? row[k] : 0.0f;
        amax = std::max(amax, std::fabs(v));
      }
      uint16_t dh = fp32_to_fp16(amax > 0.0f ? (amax / 127.0f) : 1.0f);
      float d = fp16_to_fp32(dh);
      ob_block_q8_0_t& blk = (*out)[static_cast<size_t>(n) * nb_k + bi];
      blk.d = dh;
      for (uint32_t j = 0; j < OB_QK8_0; ++j) {
        uint32_t k = bi * OB_QK8_0 + j;
        float v = (k < k_dim) ? row[k] : 0.0f;
        long q = (d > 0.0f) ? std::lround(v / d) : 0L;
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        blk.qs[j] = static_cast<int8_t>(q);
      }
    }
  }
}

struct ProcMapsAudit {
  bool maps_has_google_tpu_libs = false;
  std::vector<std::string> mapped_libs;
};

ProcMapsAudit audit_proc_maps() {
  ProcMapsAudit audit{};
  std::ifstream ifs("/proc/self/maps");
  if (!ifs) return audit;
  std::string line;
  while (std::getline(ifs, line)) {
    auto pos = line.find('/');
    if (pos == std::string::npos) continue;
    std::string path = line.substr(pos);
    if (path.find(".so") == std::string::npos) continue;
    if (std::find(audit.mapped_libs.begin(), audit.mapped_libs.end(), path) == audit.mapped_libs.end()) {
      audit.mapped_libs.push_back(path);
    }
    if (path.find("libedgetpu") != std::string::npos ||
        path.find("libdarwinn") != std::string::npos ||
        path.find("libtachyon") != std::string::npos ||
        path.find("libtflite") != std::string::npos ||
        path.find("libtensorflow") != std::string::npos ||
        path.find("libgxp") != std::string::npos ||
        path.find("/vendor/lib") != std::string::npos) {
      audit.maps_has_google_tpu_libs = true;
    }
  }
  return audit;
}

bool is_supported_op(const std::string& op) {
  return op == "matmul" || op == "matmul_silu" || op == "fused_matmul_silu" ||
         op == "matmul_gelu" || op == "fused_matmul_gelu" || op == "swiglu";
}

uint32_t parse_activation(const std::string& op) {
  if (op == "matmul") return OB_ACT_NONE;
  if (op == "matmul_silu" || op == "fused_matmul_silu") return OB_ACT_SILU;
  if (op == "matmul_gelu" || op == "fused_matmul_gelu") return OB_ACT_GELU;
  return OB_ACT_NONE;
}

bool parse_args(int argc, char** argv, RunnerArgs* args, std::string* err) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        *err = std::string("Missing argument for ") + flag;
        return "";
      }
      return argv[++i];
    };
    if (a == "--op") args->op = next("--op");
    else if (a == "--dtype") args->dtype = next("--dtype");
    else if (a == "--m") args->m = static_cast<uint32_t>(std::stoul(next("--m")));
    else if (a == "--k") args->k = static_cast<uint32_t>(std::stoul(next("--k")));
    else if (a == "--n") args->n = static_cast<uint32_t>(std::stoul(next("--n")));
    else if (a == "--n-inter") args->n_inter = static_cast<uint32_t>(std::stoul(next("--n-inter")));
    else if (a == "--in") args->in_path = next("--in");
    else if (a == "--w") args->w_path = next("--w");
    else if (a == "--w-up") args->w_up_path = next("--w-up");
    else if (a == "--w-down") args->w_down_path = next("--w-down");
    else if (a == "--bias") args->bias_path = next("--bias");
    else if (a == "--out") args->out_path = next("--out");
    else if (a == "--dive-interp") args->dive_interp_path = next("--dive-interp");
    else if (a == "--s-in") args->s_in = std::stof(next("--s-in"));
    else if (a == "--zp-in") args->zp_in = std::stoi(next("--zp-in"));
    else if (a == "--s-w") args->s_w = std::stof(next("--s-w"));
    else if (a == "--s-out") args->s_out = std::stof(next("--s-out"));
    else if (a == "--zp-out") args->zp_out = std::stoi(next("--zp-out"));
    else if (a == "--warmup") args->warmup = std::stoi(next("--warmup"));
    else if (a == "--iters") args->iters = std::stoi(next("--iters"));
    else if (a == "--perf-mode") args->perf_mode = std::stoi(next("--perf-mode"));
    else if (a == "--mock") args->mock = true;
    else {
      *err = "Unknown argument: " + a;
      return false;
    }
    if (!err->empty()) return false;
  }
  if (!is_supported_op(args->op)) {
    *err = "Unsupported --op: " + args->op;
    return false;
  }
  if (args->in_path.empty() || args->w_path.empty() || args->out_path.empty()) {
    *err = "--in, --w, and --out are required";
    return false;
  }
  if (args->n_inter == 0u) args->n_inter = args->n;
  return true;
}

double median_vec(std::vector<double> vals) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  size_t mid = vals.size() / 2u;
  if (vals.size() % 2u == 1u) return vals[mid];
  return 0.5 * (vals[mid - 1u] + vals[mid]);
}

}  // namespace

int main(int argc, char** argv) {
  RunnerArgs args{};
  std::string err;
  if (!parse_args(argc, argv, &args, &err)) {
    std::fprintf(stderr, "freetpu_tile_runner: %s\n", err.c_str());
    return 2;
  }

  const char* mock_env = std::getenv("OPENBUENOS_MOCK");
  if (mock_env && mock_env[0] == '1') args.mock = true;

  std::vector<uint8_t> in_bytes, w_bytes, w_up_bytes, w_down_bytes, bias_bytes;
  if (!read_bin(args.in_path, &in_bytes, &err) || !read_bin(args.w_path, &w_bytes, &err)) {
    std::fprintf(stderr, "freetpu_tile_runner: %s\n", err.c_str());
    return 1;
  }
  if (!args.bias_path.empty() && !read_bin(args.bias_path, &bias_bytes, &err)) {
    std::fprintf(stderr, "freetpu_tile_runner: %s\n", err.c_str());
    return 1;
  }
  if (args.op == "swiglu") {
    if (args.w_up_path.empty() || args.w_down_path.empty()) {
      std::fprintf(stderr, "freetpu_tile_runner: swiglu requires --w-up and --w-down\n");
      return 2;
    }
    if (!read_bin(args.w_up_path, &w_up_bytes, &err) || !read_bin(args.w_down_path, &w_down_bytes, &err)) {
      std::fprintf(stderr, "freetpu_tile_runner: %s\n", err.c_str());
      return 1;
    }
  }

  ob_config_t cfg = ob_config_default();
  cfg.mock_mode = args.mock ? 1 : 0;
  cfg.max_inflight = 16u;
  cfg.graph_cache_capacity = 256u;
  if (args.perf_mode >= 0) cfg.perf_mode = static_cast<uint32_t>(args.perf_mode);

  ob_context_t* ctx = nullptr;
  ob_status_t st = ob_context_create(&cfg, &ctx);
  if (st != OB_OK || !ctx) {
    std::fprintf(stderr, "ob_context_create failed: %s\n", ob_status_string(st));
    return 1;
  }

  bool emitter_registered = false;
  if (ob_gemm_register_builtin_emitter(ctx, args.dive_interp_path.c_str()) == OB_OK) {
    emitter_registered = true;
  } else if (!args.mock) {
    std::fprintf(stderr, "ob_gemm_register_builtin_emitter failed: %s (%s)\n",
                 ob_status_string(st), ob_context_last_error(ctx));
    ob_context_destroy(ctx);
    return 1;
  }

  std::vector<double> hw_us_samples;
  std::vector<double> wall_us_samples;
  uint32_t tiles_dispatched = 0u;
  uint32_t cache_hits = 0u;
  size_t dgc0_bytes = 0u;
  bool ok = true;

  if (args.dtype == "int8") {
    if (args.op == "swiglu") {
      std::fprintf(stderr, "int8 swiglu requires float/w8a16 path in runner\n");
      ob_context_destroy(ctx);
      return 2;
    }
    const size_t expected_in = static_cast<size_t>(args.m) * args.k;
    const size_t expected_w = static_cast<size_t>(args.n) * args.k;
    if (in_bytes.size() != expected_in || w_bytes.size() != expected_w) {
      std::fprintf(stderr, "int8 tensor size mismatch: in=%zu (exp %zu), w=%zu (exp %zu)\n",
                   in_bytes.size(), expected_in, w_bytes.size(), expected_w);
      ob_context_destroy(ctx);
      return 2;
    }

    std::vector<int32_t> bias_i32(args.n, 0);
    if (!bias_bytes.empty() && bias_bytes.size() == static_cast<size_t>(args.n) * 4u) {
      std::memcpy(bias_i32.data(), bias_bytes.data(), bias_bytes.size());
    }

    ob_gemm_desc_t desc{};
    desc.m_dim = args.m;
    desc.k_dim = args.k;
    desc.n_dim = args.n;
    desc.s_in = args.s_in;
    desc.zp_in = args.zp_in;
    desc.s_w = args.s_w;
    desc.zp_w = 0;
    desc.s_out = args.s_out;
    desc.zp_out = args.zp_out;
    desc.activation = parse_activation(args.op);
    desc.weights_nk = reinterpret_cast<const int8_t*>(w_bytes.data());
    desc.bias_n = bias_i32.data();

    size_t req_sz = 0u;
    st = ob_gemm_synthesize_dgc0(ctx, &desc, nullptr, 0u, &req_sz);
    std::vector<uint8_t> dgc0(req_sz, 0u);
    if (st == OB_ERR_OVERFLOW && req_sz > 0u) {
      st = ob_gemm_synthesize_dgc0(ctx, &desc, dgc0.data(), dgc0.size(), &req_sz);
    }
    if (st != OB_OK) {
      std::fprintf(stderr, "ob_gemm_synthesize_dgc0 failed: %s (%s)\n",
                   ob_status_string(st), ob_context_last_error(ctx));
      ob_context_destroy(ctx);
      return 1;
    }
    dgc0_bytes = req_sz;

    ob_graph_t* g = nullptr;
    st = ob_graph_register_dgc0(ctx, dgc0.data(), dgc0.size(), 0u, &g);
    ob_buffer_t* in_buf = nullptr;
    ob_buffer_t* out_buf = nullptr;
    if (st == OB_OK) st = ob_buffer_alloc(ctx, std::max<size_t>(expected_in, 4096u), OB_BUF_TO_DEVICE | OB_BUF_ZERO_INIT, &in_buf);
    if (st == OB_OK) st = ob_buffer_alloc(ctx, std::max<size_t>(static_cast<size_t>(args.m) * args.n, 4096u), OB_BUF_FROM_DEVICE | OB_BUF_ZERO_INIT, &out_buf);
    if (st == OB_OK) st = ob_buffer_write(in_buf, in_bytes.data(), expected_in, 0u);

    for (int i = 0; st == OB_OK && i < args.warmup; ++i) {
      ob_invoke_stats_t ws{};
      st = ob_graph_invoke(g, &in_buf, 1u, &out_buf, 1u, &ws);
    }
    for (int i = 0; st == OB_OK && i < std::max(1, args.iters); ++i) {
      ob_invoke_stats_t is{};
      auto t0 = std::chrono::steady_clock::now();
      st = ob_graph_invoke(g, &in_buf, 1u, &out_buf, 1u, &is);
      auto t1 = std::chrono::steady_clock::now();
      double wall_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      hw_us_samples.push_back(static_cast<double>(is.hw_execution_us));
      wall_us_samples.push_back(wall_us);
      tiles_dispatched = 1u;
    }

    std::vector<uint8_t> out_data(static_cast<size_t>(args.m) * args.n, 0u);
    if (st == OB_OK) {
      st = ob_buffer_read(out_buf, out_data.data(), out_data.size(), 0u);
    }
    if (st != OB_OK || !write_bin(args.out_path, out_data.data(), out_data.size(), &err)) {
      ok = false;
    }
    if (out_buf) ob_buffer_free(out_buf);
    if (in_buf) ob_buffer_free(in_buf);
    if (g) ob_graph_release(g);
  } else {
    // w8a16 / bf16 path using Q8_0 weights + FP32 activations via ob_gemm_matmul_f32_quant_batch
    const uint32_t k_pad = ((args.k + OB_QK8_0 - 1u) / OB_QK8_0) * OB_QK8_0;
    const size_t expected_in = static_cast<size_t>(args.m) * args.k * sizeof(float);
    if (in_bytes.size() != expected_in) {
      std::fprintf(stderr, "f32 input size mismatch: %zu vs expected %zu\n", in_bytes.size(), expected_in);
      ob_context_destroy(ctx);
      return 2;
    }
    std::vector<float> a_padded(static_cast<size_t>(args.m) * k_pad, 0.0f);
    const float* a_src = reinterpret_cast<const float*>(in_bytes.data());
    for (uint32_t m = 0; m < args.m; ++m) {
      std::memcpy(a_padded.data() + static_cast<size_t>(m) * k_pad,
                  a_src + static_cast<size_t>(m) * args.k,
                  static_cast<size_t>(args.k) * sizeof(float));
    }

    auto load_q8 = [&](const std::vector<uint8_t>& raw, uint32_t n_dim, uint32_t k_dim,
                       uint32_t kp, std::vector<ob_block_q8_0_t>* q8) -> bool {
      const size_t f32_sz = static_cast<size_t>(n_dim) * k_dim * sizeof(float);
      if (raw.size() != f32_sz) return false;
      quantize_f32_to_q8_0(reinterpret_cast<const float*>(raw.data()), n_dim, k_dim, kp, q8);
      return true;
    };

    if (args.op == "swiglu") {
      const uint32_t inter_pad = ((args.n_inter + OB_QK8_0 - 1u) / OB_QK8_0) * OB_QK8_0;
      std::vector<ob_block_q8_0_t> q_gate, q_up, q_down;
      if (!load_q8(w_bytes, args.n_inter, args.k, k_pad, &q_gate) ||
          !load_q8(w_up_bytes, args.n_inter, args.k, k_pad, &q_up) ||
          !load_q8(w_down_bytes, args.n, args.n_inter, inter_pad, &q_down)) {
        std::fprintf(stderr, "swiglu weight byte size mismatch\n");
        ob_context_destroy(ctx);
        return 2;
      }
      std::vector<float> c_gate(static_cast<size_t>(args.m) * args.n_inter, 0.0f);
      std::vector<float> c_up(static_cast<size_t>(args.m) * args.n_inter, 0.0f);
      std::vector<float> hidden(static_cast<size_t>(args.m) * inter_pad, 0.0f);
      std::vector<float> c_out(static_cast<size_t>(args.m) * args.n, 0.0f);

      ob_gemm_f32_quant_job_t gu_jobs[2]{};
      gu_jobs[0] = {q_gate.data(), c_gate.data(), args.n_inter, OB_ACT_SILU, 0u, {0, 0, 0}};
      gu_jobs[1] = {q_up.data(), c_up.data(), args.n_inter, OB_ACT_NONE, 0u, {0, 0, 0}};
      ob_gemm_f32_quant_job_t down_job{q_down.data(), c_out.data(), args.n, OB_ACT_NONE, 0u, {0, 0, 0}};

      auto run_once = [&](ob_invoke_stats_t* s_tot) -> ob_status_t {
        ob_invoke_stats_t s1{}, s2{};
        ob_status_t r = ob_gemm_matmul_f32_quant_batch(ctx, a_padded.data(), args.m, k_pad, gu_jobs, 2u, &s1);
        if (r != OB_OK) return r;
        for (uint32_t m = 0; m < args.m; ++m) {
          for (uint32_t j = 0; j < args.n_inter; ++j) {
            hidden[static_cast<size_t>(m) * inter_pad + j] =
                c_gate[static_cast<size_t>(m) * args.n_inter + j] *
                c_up[static_cast<size_t>(m) * args.n_inter + j];
          }
        }
        r = ob_gemm_matmul_f32_quant_batch(ctx, hidden.data(), args.m, inter_pad, &down_job, 1u, &s2);
        if (s_tot) {
          s_tot->hw_execution_us = s1.hw_execution_us + s2.hw_execution_us;
          s_tot->tiles_dispatched = s1.tiles_dispatched + s2.tiles_dispatched;
          s_tot->cache_hits = s1.cache_hits + s2.cache_hits;
        }
        return r;
      };

      for (int i = 0; st == OB_OK && i < args.warmup; ++i) {
        st = run_once(nullptr);
      }
      for (int i = 0; st == OB_OK && i < std::max(1, args.iters); ++i) {
        ob_invoke_stats_t is{};
        auto t0 = std::chrono::steady_clock::now();
        st = run_once(&is);
        auto t1 = std::chrono::steady_clock::now();
        hw_us_samples.push_back(static_cast<double>(is.hw_execution_us));
        wall_us_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        tiles_dispatched = is.tiles_dispatched;
        cache_hits = is.cache_hits;
      }
      if (st != OB_OK || !write_bin(args.out_path, c_out.data(), c_out.size() * sizeof(float), &err)) {
        ok = false;
      }
    } else {
      std::vector<ob_block_q8_0_t> q_w;
      if (!load_q8(w_bytes, args.n, args.k, k_pad, &q_w)) {
        std::fprintf(stderr, "w8a16/bf16 weight byte size mismatch\n");
        ob_context_destroy(ctx);
        return 2;
      }
      std::vector<float> c_out(static_cast<size_t>(args.m) * args.n, 0.0f);
      ob_gemm_f32_quant_job_t job{q_w.data(), c_out.data(), args.n, parse_activation(args.op), 0u, {0, 0, 0}};
      for (int i = 0; st == OB_OK && i < args.warmup; ++i) {
        st = ob_gemm_matmul_f32_quant_batch(ctx, a_padded.data(), args.m, k_pad, &job, 1u, nullptr);
      }
      for (int i = 0; st == OB_OK && i < std::max(1, args.iters); ++i) {
        ob_invoke_stats_t is{};
        auto t0 = std::chrono::steady_clock::now();
        st = ob_gemm_matmul_f32_quant_batch(ctx, a_padded.data(), args.m, k_pad, &job, 1u, &is);
        auto t1 = std::chrono::steady_clock::now();
        hw_us_samples.push_back(static_cast<double>(is.hw_execution_us));
        wall_us_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        tiles_dispatched = is.tiles_dispatched;
        cache_hits = is.cache_hits;
      }
      if (st != OB_OK || !write_bin(args.out_path, c_out.data(), c_out.size() * sizeof(float), &err)) {
        ok = false;
      }
    }
  }

  ProcMapsAudit maps_audit = audit_proc_maps();
  double med_hw_us = median_vec(hw_us_samples);
  double min_hw_us = hw_us_samples.empty() ? 0.0 : *std::min_element(hw_us_samples.begin(), hw_us_samples.end());
  double med_wall_us = median_vec(wall_us_samples);
  double min_wall_us = wall_us_samples.empty() ? 0.0 : *std::min_element(wall_us_samples.begin(), wall_us_samples.end());

  std::ostringstream js;
  js << "{\n"
     << "  \"ok\": " << ((ok && st == OB_OK && !maps_audit.maps_has_google_tpu_libs) ? "true" : "false") << ",\n"
     << "  \"op\": " << json_str(args.op) << ",\n"
     << "  \"dtype\": " << json_str(args.dtype) << ",\n"
     << "  \"m\": " << args.m << ",\n"
     << "  \"k\": " << args.k << ",\n"
     << "  \"n\": " << args.n << ",\n"
     << "  \"n_inter\": " << args.n_inter << ",\n"
     << "  \"mock\": " << (ob_context_is_mock(ctx) ? "true" : "false") << ",\n"
     << "  \"emitter_registered\": " << (emitter_registered ? "true" : "false") << ",\n"
     << "  \"perf_mode\": " << args.perf_mode << ",\n"
     << "  \"iters\": " << args.iters << ",\n"
     << "  \"tiles_dispatched\": " << tiles_dispatched << ",\n"
     << "  \"cache_hits\": " << cache_hits << ",\n"
     << "  \"dgc0_bytes\": " << dgc0_bytes << ",\n"
     << "  \"hw_us\": " << med_hw_us << ",\n"
     << "  \"min_hw_us\": " << min_hw_us << ",\n"
     << "  \"wall_us\": " << med_wall_us << ",\n"
     << "  \"min_wall_us\": " << min_wall_us << ",\n"
     << "  \"maps_has_google_tpu_libs\": " << (maps_audit.maps_has_google_tpu_libs ? "true" : "false") << ",\n"
     << "  \"status\": " << json_str(ob_status_string(st)) << "\n"
     << "}\n";
  std::printf("%s", js.str().c_str());

  ob_context_destroy(ctx);
  return (ok && st == OB_OK && !maps_audit.maps_has_google_tpu_libs) ? 0 : 1;
}
