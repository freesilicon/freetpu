#include "stablehlo_emitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cloudtpu {

std::string format_mlir_tensor_type(PjrtDType dtype, const std::vector<int64_t>& dims) {
    std::ostringstream oss;
    oss << "tensor<";
    for (int64_t d : dims) {
        oss << d << "x";
    }
    oss << pjrt_dtype_mlir_name(dtype) << ">";
    return oss.str();
}

namespace {

struct SsaValue {
    std::string name;
    PjrtDType dtype = PjrtDType::F32;
    std::vector<int64_t> dims;

    std::string mlir_type() const { return format_mlir_tensor_type(dtype, dims); }
};

std::string format_f32_literal(float v) {
    if (!std::isfinite(v)) {
        v = (v < 0.0f) ? -3.40282347e+38f : 3.40282347e+38f;
    }
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(8) << v;
    return oss.str();
}

std::string format_dims_bracket(const std::vector<int64_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

class StableHloBuilder {
public:
    void add_arg(const std::string& arg_name, PjrtDType dtype, const std::vector<int64_t>& dims) {
        args_.push_back({arg_name, dtype, dims});
    }

    SsaValue arg_val(size_t idx) const { return args_.at(idx); }

    SsaValue emit_const_splat_f32(float val, const std::vector<int64_t>& dims, PjrtDType dt = PjrtDType::F32) {
        SsaValue out = next_val(dt, dims);
        body_ << "    " << out.name << " = stablehlo.constant dense<" << format_f32_literal(val)
              << "> : " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_const_splat_int(int64_t val, const std::vector<int64_t>& dims, PjrtDType dt) {
        SsaValue out = next_val(dt, dims);
        body_ << "    " << out.name << " = stablehlo.constant dense<" << val << "> : "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_const_1d_f32(const std::vector<float>& vals, const std::vector<int64_t>& dims) {
        const std::vector<int64_t> flat_dims = {static_cast<int64_t>(vals.size())};
        SsaValue flat = next_val(PjrtDType::F32, flat_dims);
        body_ << "    " << flat.name << " = stablehlo.constant dense<[";
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) body_ << ", ";
            body_ << format_f32_literal(vals[i]);
        }
        body_ << "]> : " << flat.mlir_type() << "\n";
        return emit_reshape(flat, dims);
    }

    SsaValue emit_iota(int64_t dim, const std::vector<int64_t>& dims, PjrtDType dt = PjrtDType::S32) {
        SsaValue out = next_val(dt, dims);
        body_ << "    " << out.name << " = stablehlo.iota dim = " << dim << " : "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_convert(const SsaValue& x, PjrtDType dst_dt) {
        if (x.dtype == dst_dt) {
            return x;
        }
        SsaValue out = next_val(dst_dt, x.dims);
        body_ << "    " << out.name << " = stablehlo.convert " << x.name << " : ("
              << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_binary(const char* op_name, const SsaValue& a, const SsaValue& b) {
        SsaValue out = next_val(a.dtype, a.dims);
        body_ << "    " << out.name << " = " << op_name << " " << a.name << ", " << b.name
              << " : " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_unary(const char* op_name, const SsaValue& a) {
        SsaValue out = next_val(a.dtype, a.dims);
        body_ << "    " << out.name << " = " << op_name << " " << a.name << " : "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_reshape(const SsaValue& x, const std::vector<int64_t>& new_dims) {
        if (x.dims == new_dims) {
            return x;
        }
        SsaValue out = next_val(x.dtype, new_dims);
        body_ << "    " << out.name << " = stablehlo.reshape " << x.name << " : ("
              << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_broadcast_in_dim(const SsaValue& x,
                                   const std::vector<int64_t>& bcast_dims,
                                   const std::vector<int64_t>& out_dims) {
        if (x.dims == out_dims) {
            return x;
        }
        SsaValue out = next_val(x.dtype, out_dims);
        body_ << "    " << out.name << " = stablehlo.broadcast_in_dim " << x.name
              << ", dims = " << format_dims_bracket(bcast_dims) << " : ("
              << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_slice(const SsaValue& x,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& limits) {
        std::vector<int64_t> out_dims(starts.size());
        for (size_t i = 0; i < starts.size(); ++i) {
            out_dims[i] = limits[i] - starts[i];
        }
        if (out_dims == x.dims) {
            return x;
        }
        SsaValue out = next_val(x.dtype, out_dims);
        body_ << "    " << out.name << " = stablehlo.slice " << x.name << " [";
        for (size_t i = 0; i < starts.size(); ++i) {
            if (i > 0) body_ << ", ";
            body_ << starts[i] << ":" << limits[i] << ":1";
        }
        body_ << "] : (" << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_concatenate(const std::vector<SsaValue>& inputs, int64_t dim) {
        if (inputs.size() == 1) {
            return inputs[0];
        }
        std::vector<int64_t> out_dims = inputs[0].dims;
        for (size_t i = 1; i < inputs.size(); ++i) {
            out_dims[static_cast<size_t>(dim)] += inputs[i].dims[static_cast<size_t>(dim)];
        }
        SsaValue out = next_val(inputs[0].dtype, out_dims);
        body_ << "    " << out.name << " = stablehlo.concatenate ";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) body_ << ", ";
            body_ << inputs[i].name;
        }
        body_ << ", dim = " << dim << " : (";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) body_ << ", ";
            body_ << inputs[i].mlir_type();
        }
        body_ << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_dot_general(const SsaValue& lhs,
                              const SsaValue& rhs,
                              const std::vector<int64_t>& lhs_batch,
                              const std::vector<int64_t>& rhs_batch,
                              const std::vector<int64_t>& lhs_contract,
                              const std::vector<int64_t>& rhs_contract,
                              const std::vector<int64_t>& out_dims,
                              PjrtDType out_dtype,
                              bool high_precision) {
        SsaValue out = next_val(out_dtype, out_dims);
        const char* prec = high_precision ? "HIGHEST" : "DEFAULT";
        body_ << "    " << out.name << " = stablehlo.dot_general " << lhs.name << ", " << rhs.name
              << ", batching_dims = " << format_dims_bracket(lhs_batch) << " x "
              << format_dims_bracket(rhs_batch)
              << ", contracting_dims = " << format_dims_bracket(lhs_contract) << " x "
              << format_dims_bracket(rhs_contract)
              << ", precision = [" << prec << ", " << prec << "] : ("
              << lhs.mlir_type() << ", " << rhs.mlir_type() << ") -> "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_reduce_axis(const SsaValue& x,
                              const SsaValue& init,
                              int64_t axis,
                              const char* reducer_op) {
        std::vector<int64_t> out_dims;
        for (size_t i = 0; i < x.dims.size(); ++i) {
            if (static_cast<int64_t>(i) != axis) {
                out_dims.push_back(x.dims[i]);
            }
        }
        SsaValue out = next_val(x.dtype, out_dims);
        const std::string scalar_t = format_mlir_tensor_type(x.dtype, {});
        const std::string lhs_arg = next_tmp_name("red_lhs");
        const std::string rhs_arg = next_tmp_name("red_rhs");
        const std::string acc_val = next_tmp_name("red_acc");
        body_ << "    " << out.name << " = \"stablehlo.reduce\"(" << x.name << ", " << init.name << ") ({\n"
              << "    ^bb0(" << lhs_arg << ": " << scalar_t << ", " << rhs_arg << ": " << scalar_t << "):\n"
              << "      " << acc_val << " = " << reducer_op << " " << lhs_arg << ", " << rhs_arg
              << " : " << scalar_t << "\n"
              << "      stablehlo.return " << acc_val << " : " << scalar_t << "\n"
              << "    }) {dimensions = array<i64: " << axis << ">} : ("
              << x.mlir_type() << ", " << init.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_gather_3d_rows(const SsaValue& operand, const SsaValue& start_indices, const std::vector<int64_t>& out_dims) {
        // operand: [B3, B2, R, D], start_indices: [B3, B2, N_idx, 3], out: [B3, B2, N_idx, D]
        SsaValue out = next_val(operand.dtype, out_dims);
        body_ << "    " << out.name << " = \"stablehlo.gather\"(" << operand.name << ", " << start_indices.name << ") {\n"
              << "      dimension_numbers = #stablehlo.gather<\n"
              << "        offset_dims = [3],\n"
              << "        collapsed_slice_dims = [0, 1, 2],\n"
              << "        start_index_map = [0, 1, 2],\n"
              << "        index_vector_dim = 3\n"
              << "      >,\n"
              << "      slice_sizes = array<i64: 1, 1, 1, " << operand.dims[3] << ">,\n"
              << "      indices_are_sorted = false\n"
              << "    } : (" << operand.mlir_type() << ", " << start_indices.mlir_type() << ") -> "
              << out.mlir_type() << "\n";
        return out;
    }

    std::string finish(const std::vector<SsaValue>& outputs) const {
        std::ostringstream mod;
        mod << "module @ggml_cloudtpu {\n";
        mod << "  func.func public @main(";
        for (size_t i = 0; i < args_.size(); ++i) {
            if (i > 0) mod << ", ";
            mod << args_[i].name << ": " << args_[i].mlir_type();
        }
        mod << ") -> ";
        if (outputs.size() == 1) {
            mod << outputs[0].mlir_type();
        } else {
            mod << "(";
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i > 0) mod << ", ";
                mod << outputs[i].mlir_type();
            }
            mod << ")";
        }
        mod << " {\n";
        mod << body_.str();
        mod << "    return ";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) mod << ", ";
            mod << outputs[i].name;
        }
        mod << " : ";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) mod << ", ";
            mod << outputs[i].mlir_type();
        }
        mod << "\n  }\n}\n";
        return mod.str();
    }

private:
    SsaValue next_val(PjrtDType dt, std::vector<int64_t> dims) {
        std::ostringstream oss;
        oss << "%v" << next_id_++;
        return {oss.str(), dt, std::move(dims)};
    }

    std::string next_tmp_name(const char* prefix) {
        std::ostringstream oss;
        oss << "%" << prefix << "_" << next_id_++;
        return oss.str();
    }

    std::vector<SsaValue> args_;
    std::ostringstream body_;
    int next_id_ = 0;
};

SsaValue dequantize_split_weight(StableHloBuilder& b,
                                 TensorStorageKind storage,
                                 const SsaValue& qs,
                                 const SsaValue& d,
                                 PjrtDType out_dt) {
    // qs: [B3, B2, N, NB, 32] S8 (Q8_0) or [B3, B2, N, NB, 16] U8 (Q4_0)
    // d:  [B3, B2, N, NB] F16
    const int64_t B3 = qs.dims[0];
    const int64_t B2 = qs.dims[1];
    const int64_t N = qs.dims[2];
    const int64_t NB = qs.dims[3];
    const int64_t K = NB * 32;
    const std::vector<int64_t> blk_dims = {B3, B2, N, NB, 32};

    SsaValue q_f;
    if (storage == TensorStorageKind::Q8_0_SPLIT) {
        q_f = b.emit_convert(qs, out_dt);
    } else if (storage == TensorStorageKind::Q4_0_SPLIT) {
        const std::vector<int64_t> half_dims = {B3, B2, N, NB, 16};
        SsaValue mask = b.emit_const_splat_int(0x0F, half_dims, PjrtDType::U8);
        SsaValue shift4 = b.emit_const_splat_int(4, half_dims, PjrtDType::U8);
        SsaValue lo_u8 = b.emit_binary("stablehlo.and", qs, mask);
        SsaValue hi_u8 = b.emit_binary("stablehlo.shift_right_logical", qs, shift4);
        SsaValue cat_u8 = b.emit_concatenate({lo_u8, hi_u8}, 4);
        SsaValue cat_i32 = b.emit_convert(cat_u8, PjrtDType::S32);
        SsaValue eight = b.emit_const_splat_int(8, blk_dims, PjrtDType::S32);
        SsaValue signed_i32 = b.emit_binary("stablehlo.subtract", cat_i32, eight);
        q_f = b.emit_convert(signed_i32, out_dt);
    } else {
        throw std::runtime_error("dequantize_split_weight: unexpected storage kind");
    }

    SsaValue d_c = b.emit_convert(d, out_dt);
    SsaValue d_b = b.emit_broadcast_in_dim(d_c, {0, 1, 2, 3}, blk_dims);
    SsaValue scaled = b.emit_binary("stablehlo.multiply", q_f, d_b);
    return b.emit_reshape(scaled, {B3, B2, N, K});
}

SsaValue broadcast_4d(StableHloBuilder& b, const SsaValue& x, const std::vector<int64_t>& target_dims) {
    if (x.dims == target_dims) {
        return x;
    }
    // Check if pure 1 -> N broadcast
    bool simple_one_to_n = true;
    for (size_t i = 0; i < 4; ++i) {
        if (x.dims[i] != target_dims[i] && x.dims[i] != 1) {
            simple_one_to_n = false;
            break;
        }
    }
    if (simple_one_to_n) {
        return b.emit_broadcast_in_dim(x, {0, 1, 2, 3}, target_dims);
    }

    // General GGML periodic repeat broadcast (target_dims[i] % x.dims[i] == 0)
    std::vector<int64_t> split_src(8);
    std::vector<int64_t> split_dst(8);
    for (size_t i = 0; i < 4; ++i) {
        if (target_dims[i] % x.dims[i] != 0) {
            throw std::runtime_error("StableHLO broadcast_4d: non-divisible shape");
        }
        split_src[2 * i + 0] = 1;
        split_src[2 * i + 1] = x.dims[i];
        split_dst[2 * i + 0] = target_dims[i] / x.dims[i];
        split_dst[2 * i + 1] = x.dims[i];
    }
    SsaValue r = b.emit_reshape(x, split_src);
    SsaValue br = b.emit_broadcast_in_dim(r, {0, 1, 2, 3, 4, 5, 6, 7}, split_dst);
    return b.emit_reshape(br, target_dims);
}

SsaValue emit_silu(StableHloBuilder& b, const SsaValue& x_in) {
    const PjrtDType orig_dt = x_in.dtype;
    SsaValue x = (orig_dt == PjrtDType::F32) ? x_in : b.emit_convert(x_in, PjrtDType::F32);
    SsaValue sig = b.emit_unary("stablehlo.logistic", x);
    SsaValue res = b.emit_binary("stablehlo.multiply", x, sig);
    return (orig_dt == PjrtDType::F32) ? res : b.emit_convert(res, orig_dt);
}

// Mirrors GGML's YaRN correlation dimensions and frequency scaling exactly.
void compute_rope_yarn_tables(const RopeParams& rp, std::vector<float>& out_inv_freq, float& out_mscale) {
    const int half = rp.n_dims / 2;
    out_inv_freq.resize(static_cast<size_t>(half));

    auto yarn_corr_dim = [&](float n_rot) -> float {
        return static_cast<float>(rp.n_dims) *
               std::log(static_cast<float>(rp.n_ctx_orig) / (n_rot * 2.0f * 3.14159265358979323846f)) /
               (2.0f * std::log(rp.freq_base));
    };

    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (rp.ext_factor != 0.0f) {
        float start = yarn_corr_dim(rp.beta_fast);
        float end = yarn_corr_dim(rp.beta_slow);
        if (rp.truncate) {
            start = std::floor(start);
            end = std::ceil(end);
        }
        corr0 = std::max(0.0f, start);
        corr1 = std::min(static_cast<float>(rp.n_dims - 1), end);
    }

    float mscale = rp.attn_factor;
    if (rp.ext_factor != 0.0f) {
        mscale *= 1.0f + 0.1f * std::log(1.0f / rp.freq_scale);
    }
    out_mscale = mscale;

    const float theta_scale = std::pow(rp.freq_base, -2.0f / static_cast<float>(rp.n_dims));
    float theta_extrap = 1.0f;
    for (int i = 0; i < half; ++i) {
        const float theta_interp = rp.freq_scale * theta_extrap;
        float theta = theta_interp;
        if (rp.ext_factor != 0.0f) {
            const float y = (static_cast<float>(i) - corr0) / std::max(0.001f, corr1 - corr0);
            const float ramp_mix = (1.0f - std::min(1.0f, std::max(0.0f, y))) * rp.ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        }
        out_inv_freq[static_cast<size_t>(i)] = theta;
        theta_extrap *= theta_scale;
    }
}

uint64_t fnv1a_64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string build_canonical_cache_key(const std::string& mlir_text) {
    // Structural key combining size + two independent FNV hashes (64-bit normal + 64-bit salted)
    // and exact length to prevent collisions without external SHA libraries.
    const uint64_t h1 = fnv1a_64(mlir_text);
    const uint64_t h2 = fnv1a_64(std::string("cloudtpu_v1:") + mlir_text);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h1
        << std::setw(16) << h2 << "_" << std::dec << mlir_text.size();
    return oss.str();
}

}  // namespace

EmittedProgram emit_stablehlo_program(const GraphProgramSpec& spec) {
    EmittedProgram prog;
    StableHloBuilder builder;

    struct TensorHandle {
        TensorStorageKind storage = TensorStorageKind::DENSE;
        SsaValue dense;
        SsaValue q_qs;
        SsaValue q_d;
    };

    std::unordered_map<uint64_t, TensorHandle> env;

    size_t arg_idx = 0;
    for (const TensorSpec& in : spec.inputs) {
        TensorHandle th;
        th.storage = in.storage;
        const int64_t B3 = in.dims[0];
        const int64_t B2 = in.dims[1];
        const int64_t M_or_N = in.dims[2];
        const int64_t K = in.dims[3];

        if (in.storage == TensorStorageKind::DENSE) {
            std::vector<int64_t> dims = {B3, B2, M_or_N, K};
            const std::string aname = "%arg" + std::to_string(arg_idx++);
            builder.add_arg(aname, in.dtype, dims);
            th.dense = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 0, in.dtype, std::move(dims)});
        } else {
            if (K <= 0 || (K % 32) != 0) {
                throw std::runtime_error("emit_stablehlo_program: quantized tensor K must be divisible by 32");
            }
            const int64_t NB = K / 32;
            const PjrtDType q_dt = (in.storage == TensorStorageKind::Q8_0_SPLIT) ? PjrtDType::S8 : PjrtDType::U8;
            const int64_t q_inner = (in.storage == TensorStorageKind::Q8_0_SPLIT) ? 32 : 16;

            std::vector<int64_t> qs_dims = {B3, B2, M_or_N, NB, q_inner};
            const std::string q_name = "%arg" + std::to_string(arg_idx++);
            builder.add_arg(q_name, q_dt, qs_dims);
            th.q_qs = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 0, q_dt, std::move(qs_dims)});

            std::vector<int64_t> d_dims = {B3, B2, M_or_N, NB};
            const std::string d_name = "%arg" + std::to_string(arg_idx++);
            builder.add_arg(d_name, PjrtDType::F16, d_dims);
            th.q_d = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 1, PjrtDType::F16, std::move(d_dims)});
        }
        env[in.id] = std::move(th);
    }

    auto resolve_dense = [&](uint64_t id, PjrtDType dequant_dt = PjrtDType::BF16) -> SsaValue {
        const TensorHandle& th = env.at(id);
        if (th.storage == TensorStorageKind::DENSE) {
            return th.dense;
        }
        return dequantize_split_weight(builder, th.storage, th.q_qs, th.q_d, dequant_dt);
    };

    for (const OpNodeSpec& op : spec.ops) {
        const std::vector<int64_t> dst_dims = {op.dst.dims[0], op.dst.dims[1], op.dst.dims[2], op.dst.dims[3]};
        SsaValue result;

        switch (op.kind) {
            case CloudOpKind::MUL_MAT: {
                SsaValue x = resolve_dense(op.src_ids.at(1), PjrtDType::F32);
                const PjrtDType deq_dt =
                    (op.high_precision || x.dtype == PjrtDType::F32) ? PjrtDType::F32 : PjrtDType::BF16;
                SsaValue w = resolve_dense(op.src_ids.at(0), deq_dt);

                if (w.dtype != x.dtype) {
                    const PjrtDType common =
                        (w.dtype == PjrtDType::F32 || x.dtype == PjrtDType::F32) ? PjrtDType::F32 : PjrtDType::BF16;
                    w = builder.emit_convert(w, common);
                    x = builder.emit_convert(x, common);
                }

                // Broadcast batch dims [B3_w, B2_w] -> [B3_x, B2_x] when GQA/batched
                if (w.dims[0] != x.dims[0] || w.dims[1] != x.dims[1]) {
                    const int64_t r3 = x.dims[0] / w.dims[0];
                    const int64_t r2 = x.dims[1] / w.dims[1];
                    SsaValue wr = builder.emit_reshape(w, {w.dims[0], 1, w.dims[1], 1, w.dims[2], w.dims[3]});
                    SsaValue wb = builder.emit_broadcast_in_dim(
                        wr, {0, 1, 2, 3, 4, 5}, {w.dims[0], r3, w.dims[1], r2, w.dims[2], w.dims[3]});
                    w = builder.emit_reshape(wb, {x.dims[0], x.dims[1], w.dims[2], w.dims[3]});
                }

                result = builder.emit_dot_general(
                    x, w, {0, 1}, {0, 1}, {3}, {3}, dst_dims, op.dst.dtype, op.high_precision);
                break;
            }
            case CloudOpKind::RMS_NORM: {
                SsaValue x_in = resolve_dense(op.src_ids.at(0), PjrtDType::F32);
                SsaValue x = builder.emit_convert(x_in, PjrtDType::F32);
                const std::vector<int64_t> red_dims = {x.dims[0], x.dims[1], x.dims[2]};
                SsaValue sq = builder.emit_binary("stablehlo.multiply", x, x);
                SsaValue zero = builder.emit_const_splat_f32(0.0f, {}, PjrtDType::F32);
                SsaValue sum = builder.emit_reduce_axis(sq, zero, 3, "stablehlo.add");
                SsaValue inv_k = builder.emit_const_splat_f32(
                    1.0f / static_cast<float>(x.dims[3]), red_dims, PjrtDType::F32);
                SsaValue mean = builder.emit_binary("stablehlo.multiply", sum, inv_k);
                SsaValue eps = builder.emit_const_splat_f32(op.eps, red_dims, PjrtDType::F32);
                SsaValue var = builder.emit_binary("stablehlo.add", mean, eps);
                SsaValue rstd = builder.emit_unary("stablehlo.rsqrt", var);
                SsaValue rstd_b = builder.emit_broadcast_in_dim(rstd, {0, 1, 2}, x.dims);
                SsaValue normed = builder.emit_binary("stablehlo.multiply", x, rstd_b);
                result = builder.emit_convert(normed, op.dst.dtype);
                break;
            }
            case CloudOpKind::ROPE: {
                SsaValue x_in = resolve_dense(op.src_ids.at(0), PjrtDType::F32);
                SsaValue pos_in = resolve_dense(op.src_ids.at(1), PjrtDType::S32);
                SsaValue x = builder.emit_convert(x_in, PjrtDType::F32);

                const int64_t B3 = x.dims[0];
                const int64_t B2 = x.dims[1];
                const int64_t M = x.dims[2];
                const int64_t ne0 = x.dims[3];
                const int64_t n_dims = op.rope.n_dims;
                const int64_t n_offs = op.rope.n_offs;
                const int64_t half = n_dims / 2;

                std::vector<float> inv_freq;
                float mscale = 1.0f;
                compute_rope_yarn_tables(op.rope, inv_freq, mscale);

                SsaValue inv_freq_c = builder.emit_const_1d_f32(inv_freq, {1, half});
                if (op.src_ids.size() > 2 && op.src_ids[2] != 0) {
                    SsaValue ff_raw = resolve_dense(op.src_ids[2], PjrtDType::F32);
                    SsaValue ff_f32 = builder.emit_convert(ff_raw, PjrtDType::F32);
                    SsaValue ff_slice = builder.emit_slice(ff_f32, {0, 0, 0, 0}, {1, 1, 1, half});
                    SsaValue ff_2d = builder.emit_reshape(ff_slice, {1, half});
                    inv_freq_c = builder.emit_binary("stablehlo.divide", inv_freq_c, ff_2d);
                }

                // pos_in: [1, 1, 1, B2] -> [B2, 1] -> f32 -> broadcast [B2, half]
                SsaValue pos_2d = builder.emit_reshape(pos_in, {B2, 1});
                SsaValue pos_f32 = builder.emit_convert(pos_2d, PjrtDType::F32);
                SsaValue pos_b = builder.emit_broadcast_in_dim(pos_f32, {0, 1}, {B2, half});
                SsaValue freq_b = builder.emit_broadcast_in_dim(inv_freq_c, {0, 1}, {B2, half});
                SsaValue angles = builder.emit_binary("stablehlo.multiply", pos_b, freq_b);

                SsaValue cos_t = builder.emit_unary("stablehlo.cosine", angles);
                SsaValue sin_t = builder.emit_unary("stablehlo.sine", angles);
                if (mscale != 1.0f) {
                    SsaValue ms_c = builder.emit_const_splat_f32(mscale, {B2, half}, PjrtDType::F32);
                    cos_t = builder.emit_binary("stablehlo.multiply", cos_t, ms_c);
                    sin_t = builder.emit_binary("stablehlo.multiply", sin_t, ms_c);
                }

                const std::vector<int64_t> rot_half_dims = {B3, B2, M, half};
                SsaValue cos_4d = builder.emit_broadcast_in_dim(
                    builder.emit_reshape(cos_t, {1, B2, 1, half}), {0, 1, 2, 3}, rot_half_dims);
                SsaValue sin_4d = builder.emit_broadcast_in_dim(
                    builder.emit_reshape(sin_t, {1, B2, 1, half}), {0, 1, 2, 3}, rot_half_dims);

                SsaValue x_rot = builder.emit_slice(x, {0, 0, 0, n_offs}, {B3, B2, M, n_offs + n_dims});

                SsaValue x0, x1;
                const bool is_neox = (op.rope.mode == 2);
                if (!is_neox) {
                    SsaValue x_pairs = builder.emit_reshape(x_rot, {B3, B2, M, half, 2});
                    x0 = builder.emit_reshape(
                        builder.emit_slice(x_pairs, {0, 0, 0, 0, 0}, {B3, B2, M, half, 1}), rot_half_dims);
                    x1 = builder.emit_reshape(
                        builder.emit_slice(x_pairs, {0, 0, 0, 0, 1}, {B3, B2, M, half, 2}), rot_half_dims);
                } else {
                    x0 = builder.emit_slice(x_rot, {0, 0, 0, 0}, {B3, B2, M, half});
                    x1 = builder.emit_slice(x_rot, {0, 0, 0, half}, {B3, B2, M, n_dims});
                }

                SsaValue x0_cos = builder.emit_binary("stablehlo.multiply", x0, cos_4d);
                SsaValue x1_sin = builder.emit_binary("stablehlo.multiply", x1, sin_4d);
                SsaValue y0 = builder.emit_binary("stablehlo.subtract", x0_cos, x1_sin);

                SsaValue x0_sin = builder.emit_binary("stablehlo.multiply", x0, sin_4d);
                SsaValue x1_cos = builder.emit_binary("stablehlo.multiply", x1, cos_4d);
                SsaValue y1 = builder.emit_binary("stablehlo.add", x0_sin, x1_cos);

                SsaValue y_rot;
                if (!is_neox) {
                    SsaValue y0_5d = builder.emit_reshape(y0, {B3, B2, M, half, 1});
                    SsaValue y1_5d = builder.emit_reshape(y1, {B3, B2, M, half, 1});
                    SsaValue y_pairs = builder.emit_concatenate({y0_5d, y1_5d}, 4);
                    y_rot = builder.emit_reshape(y_pairs, {B3, B2, M, n_dims});
                } else {
                    y_rot = builder.emit_concatenate({y0, y1}, 3);
                }

                std::vector<SsaValue> parts;
                if (n_offs > 0) {
                    parts.push_back(builder.emit_slice(x, {0, 0, 0, 0}, {B3, B2, M, n_offs}));
                }
                parts.push_back(y_rot);
                if (n_offs + n_dims < ne0) {
                    parts.push_back(builder.emit_slice(x, {0, 0, 0, n_offs + n_dims}, {B3, B2, M, ne0}));
                }
                SsaValue out_f32 = builder.emit_concatenate(parts, 3);
                result = builder.emit_convert(out_f32, op.dst.dtype);
                break;
            }
            case CloudOpKind::SILU: {
                SsaValue x = resolve_dense(op.src_ids.at(0), PjrtDType::F32);
                result = builder.emit_convert(emit_silu(builder, x), op.dst.dtype);
                break;
            }
            case CloudOpKind::SWIGLU: {
                SsaValue gate, up;
                if (op.src_ids.size() >= 2 && op.src_ids[1] != 0) {
                    gate = resolve_dense(op.src_ids[0], PjrtDType::F32);
                    up = resolve_dense(op.src_ids[1], PjrtDType::F32);
                } else {
                    SsaValue packed = resolve_dense(op.src_ids.at(0), PjrtDType::F32);
                    const int64_t nc = packed.dims[3] / 2;
                    SsaValue h0 = builder.emit_slice(packed, {0, 0, 0, 0}, {packed.dims[0], packed.dims[1], packed.dims[2], nc});
                    SsaValue h1 = builder.emit_slice(packed, {0, 0, 0, nc}, {packed.dims[0], packed.dims[1], packed.dims[2], 2 * nc});
                    gate = op.glu_swapped ? h1 : h0;
                    up = op.glu_swapped ? h0 : h1;
                }
                SsaValue g_f = builder.emit_convert(gate, PjrtDType::F32);
                SsaValue u_f = builder.emit_convert(up, PjrtDType::F32);
                SsaValue prod = builder.emit_binary("stablehlo.multiply", emit_silu(builder, g_f), u_f);
                result = builder.emit_convert(prod, op.dst.dtype);
                break;
            }
            case CloudOpKind::SOFT_MAX: {
                SsaValue x_in = resolve_dense(op.src_ids.at(0), PjrtDType::F32);
                SsaValue s = builder.emit_convert(x_in, PjrtDType::F32);
                if (op.soft_max.scale != 1.0f) {
                    SsaValue sc = builder.emit_const_splat_f32(op.soft_max.scale, s.dims, PjrtDType::F32);
                    s = builder.emit_binary("stablehlo.multiply", s, sc);
                }
                if (op.src_ids.size() > 1 && op.src_ids[1] != 0) {
                    SsaValue m_raw = resolve_dense(op.src_ids[1], PjrtDType::F32);
                    SsaValue m = builder.emit_convert(m_raw, PjrtDType::F32);
                    if (m.dims[2] > s.dims[2]) {
                        m = builder.emit_slice(m, {0, 0, 0, 0}, {m.dims[0], m.dims[1], s.dims[2], m.dims[3]});
                    }
                    m = broadcast_4d(builder, m, s.dims);
                    if (op.soft_max.max_bias > 0.0f) {
                        const uint32_t n_head = static_cast<uint32_t>(s.dims[1]);
                        const uint32_t n_head_log2 = 1U << static_cast<uint32_t>(std::floor(std::log2(n_head)));
                        const float m0 = std::pow(2.0f, -op.soft_max.max_bias / static_cast<float>(n_head_log2));
                        const float m1 = std::pow(2.0f, -(op.soft_max.max_bias * 0.5f) / static_cast<float>(n_head_log2));
                        std::vector<float> slopes(n_head);
                        for (uint32_t h = 0; h < n_head; ++h) {
                            slopes[h] = (h < n_head_log2)
                                            ? std::pow(m0, static_cast<float>(h + 1))
                                            : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
                        }
                        SsaValue sc = builder.emit_const_1d_f32(slopes, {1, static_cast<int64_t>(n_head), 1, 1});
                        SsaValue sc_b = builder.emit_broadcast_in_dim(sc, {0, 1, 2, 3}, s.dims);
                        m = builder.emit_binary("stablehlo.multiply", m, sc_b);
                    }
                    s = builder.emit_binary("stablehlo.add", s, m);
                }

                SsaValue neg_inf = builder.emit_const_splat_f32(-3.40282347e+38f, {}, PjrtDType::F32);
                SsaValue rmax = builder.emit_reduce_axis(s, neg_inf, 3, "stablehlo.maximum");
                SsaValue rmax_b = builder.emit_broadcast_in_dim(rmax, {0, 1, 2}, s.dims);
                SsaValue shifted = builder.emit_binary("stablehlo.subtract", s, rmax_b);
                SsaValue exps = builder.emit_unary("stablehlo.exponential", shifted);
                SsaValue zero = builder.emit_const_splat_f32(0.0f, {}, PjrtDType::F32);
                SsaValue rsum = builder.emit_reduce_axis(exps, zero, 3, "stablehlo.add");
                SsaValue rsum_b = builder.emit_broadcast_in_dim(rsum, {0, 1, 2}, s.dims);
                SsaValue probs = builder.emit_binary("stablehlo.divide", exps, rsum_b);
                result = builder.emit_convert(probs, op.dst.dtype);
                break;
            }
            case CloudOpKind::GET_ROWS: {
                const TensorHandle& tbl = env.at(op.src_ids.at(0));
                SsaValue idx_raw = resolve_dense(op.src_ids.at(1), PjrtDType::S32);
                // idx_raw: [1, B3, B2, N_idx] -> [B3, B2, N_idx, 1]
                const int64_t B3 = dst_dims[0];
                const int64_t B2 = dst_dims[1];
                const int64_t N_idx = dst_dims[2];
                const std::vector<int64_t> idx_col_dims = {B3, B2, N_idx, 1};
                SsaValue idx3 = builder.emit_iota(0, idx_col_dims, PjrtDType::S32);
                SsaValue idx2 = builder.emit_iota(1, idx_col_dims, PjrtDType::S32);
                SsaValue idx1 = builder.emit_reshape(idx_raw, idx_col_dims);
                SsaValue start_indices = builder.emit_concatenate({idx3, idx2, idx1}, 3);

                if (tbl.storage == TensorStorageKind::DENSE) {
                    SsaValue gathered = builder.emit_gather_3d_rows(tbl.dense, start_indices, dst_dims);
                    result = builder.emit_convert(gathered, op.dst.dtype);
                } else {
                    // Gather quantized blocks first, then dequantize only the selected N_idx rows!
                    const int64_t NB = tbl.q_d.dims[3];
                    const int64_t q_inner = tbl.q_qs.dims[4];
                    SsaValue qs_flat = builder.emit_reshape(
                        tbl.q_qs, {tbl.q_qs.dims[0], tbl.q_qs.dims[1], tbl.q_qs.dims[2], NB * q_inner});
                    SsaValue g_qs_flat = builder.emit_gather_3d_rows(qs_flat, start_indices, {B3, B2, N_idx, NB * q_inner});
                    SsaValue g_qs = builder.emit_reshape(g_qs_flat, {B3, B2, N_idx, NB, q_inner});
                    SsaValue g_d = builder.emit_gather_3d_rows(tbl.q_d, start_indices, {B3, B2, N_idx, NB});
                    result = dequantize_split_weight(builder, tbl.storage, g_qs, g_d, op.dst.dtype);
                }
                break;
            }
            case CloudOpKind::ADD:
            case CloudOpKind::MUL: {
                SsaValue a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                SsaValue b = resolve_dense(op.src_ids.at(1), op.dst.dtype);
                SsaValue ab = broadcast_4d(builder, a, dst_dims);
                SsaValue bb = broadcast_4d(builder, b, dst_dims);
                const PjrtDType calc_dt = (op.dst.dtype == PjrtDType::F32) ? PjrtDType::F32 : PjrtDType::F32;
                SsaValue af = builder.emit_convert(ab, calc_dt);
                SsaValue bf = builder.emit_convert(bb, calc_dt);
                const char* mop = (op.kind == CloudOpKind::ADD) ? "stablehlo.add" : "stablehlo.multiply";
                SsaValue res_f = builder.emit_binary(mop, af, bf);
                result = builder.emit_convert(res_f, op.dst.dtype);
                break;
            }
            case CloudOpKind::CPY:
            case CloudOpKind::RESHAPE: {
                SsaValue a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                SsaValue c = builder.emit_convert(a, op.dst.dtype);
                result = builder.emit_reshape(c, dst_dims);
                break;
            }
        }

        TensorHandle out_th;
        out_th.storage = TensorStorageKind::DENSE;
        out_th.dense = std::move(result);
        env[op.dst.id] = std::move(out_th);
    }

    std::vector<SsaValue> return_vals;
    return_vals.reserve(spec.output_ids.size());
    for (uint64_t oid : spec.output_ids) {
        const SsaValue& ov = env.at(oid).dense;
        return_vals.push_back(ov);
        prog.output_bindings.push_back({oid, ov.dtype, ov.dims});
    }

    prog.mlir_text = builder.finish(return_vals);
    prog.cache_key = build_canonical_cache_key(prog.mlir_text);
    return prog;
}

}  // namespace cloudtpu
