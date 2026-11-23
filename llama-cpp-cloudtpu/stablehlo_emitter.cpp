#include "stablehlo_emitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cloudtpu {

namespace {
std::ostringstream make_classic_stream() {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    return oss;
}
}  // namespace

std::string format_mlir_tensor_type(PjrtDType dtype, const std::vector<int64_t>& dims) {
    std::ostringstream oss = make_classic_stream();
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
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        std::ostringstream oss = make_classic_stream();
        oss << "0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << bits;
        return oss.str();
    }
    std::ostringstream oss = make_classic_stream();
    oss << std::scientific << std::setprecision(8) << v;
    return oss.str();
}

std::string format_dims_bracket(const std::vector<int64_t>& dims) {
    std::ostringstream oss = make_classic_stream();
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
    StableHloBuilder() { body_.imbue(std::locale::classic()); }
    void add_arg(const std::string& arg_name,
                 PjrtDType dtype,
                 const std::vector<int64_t>& dims,
                 int32_t alias_output_idx = -1) {
        args_.push_back({arg_name, dtype, dims});
        arg_aliases_.push_back(alias_output_idx);
    }

    SsaValue arg_val(size_t idx) const { return args_.at(idx); }

    SsaValue emit_const_splat_f32(float val, const std::vector<int64_t>& dims, PjrtDType dt = PjrtDType::F32) {
        if (!std::isfinite(val) && dt != PjrtDType::F32) {
            // MLIR 32-bit IEEE hex float literals require f32; convert cleanly to target float type.
            SsaValue f32_val = emit_const_splat_f32(val, dims, PjrtDType::F32);
            return emit_convert(f32_val, dt);
        }
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

    SsaValue emit_compare(const char* dir, const SsaValue& a, const SsaValue& b) {
        SsaValue out = next_val(PjrtDType::PRED, a.dims);
        const char* cmp_type = "FLOAT";
        if (a.dtype == PjrtDType::S8 || a.dtype == PjrtDType::S16 ||
            a.dtype == PjrtDType::S32 || a.dtype == PjrtDType::S64) {
            cmp_type = "SIGNED";
        } else if (a.dtype == PjrtDType::U8 || a.dtype == PjrtDType::U32 ||
                   a.dtype == PjrtDType::PRED) {
            cmp_type = "UNSIGNED";
        }
        body_ << "    " << out.name << " = stablehlo.compare " << dir << ", " << a.name << ", " << b.name
              << ", " << cmp_type << " : (" << a.mlir_type() << ", " << b.mlir_type() << ") -> "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_select(const SsaValue& pred, const SsaValue& on_true, const SsaValue& on_false) {
        SsaValue out = next_val(on_true.dtype, on_true.dims);
        body_ << "    " << out.name << " = \"stablehlo.select\"(" << pred.name << ", " << on_true.name
              << ", " << on_false.name << ") : (" << pred.mlir_type() << ", " << on_true.mlir_type()
              << ", " << on_false.mlir_type() << ") -> " << out.mlir_type() << "\n";
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

    SsaValue emit_transpose(const SsaValue& x, const std::vector<int64_t>& perm) {
        bool is_identity = true;
        std::vector<int64_t> out_dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] != static_cast<int64_t>(i)) is_identity = false;
            out_dims[i] = x.dims[static_cast<size_t>(perm[i])];
        }
        if (is_identity) {
            return x;
        }
        SsaValue out = next_val(x.dtype, out_dims);
        body_ << "    " << out.name << " = stablehlo.transpose " << x.name
              << ", dims = " << format_dims_bracket(perm) << " : ("
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

    SsaValue emit_gather_3d_last_axis(const SsaValue& operand,
                                      const SsaValue& start_indices,
                                      const std::vector<int64_t>& out_dims) {
        // Gathers along {0, 1, last} for K-major N-minor quantized tables:
        //  - 5D [B3, B2, NB, blk, N] -> [B3, B2, N_idx, NB, blk]
        //  - 4D [B3, B2, NB, N]      -> [B3, B2, N_idx, NB]
        SsaValue out = next_val(operand.dtype, out_dims);
        const int64_t last_axis = static_cast<int64_t>(operand.dims.size()) - 1;
        body_ << "    " << out.name << " = \"stablehlo.gather\"(" << operand.name << ", " << start_indices.name << ") {\n"
              << "      dimension_numbers = #stablehlo.gather<\n"
              << "        offset_dims = [";
        for (size_t i = 3; i < out_dims.size(); ++i) {
            if (i > 3) body_ << ", ";
            body_ << i;
        }
        body_ << "],\n"
              << "        collapsed_slice_dims = [0, 1, " << last_axis << "],\n"
              << "        start_index_map = [0, 1, " << last_axis << "],\n"
              << "        index_vector_dim = 3\n"
              << "      >,\n"
              << "      slice_sizes = array<i64: 1, 1";
        for (size_t i = 2; i + 1 < operand.dims.size(); ++i) {
            body_ << ", " << operand.dims[i];
        }
        body_ << ", 1>,\n"
              << "      indices_are_sorted = false\n"
              << "    } : (" << operand.mlir_type() << ", " << start_indices.mlir_type() << ") -> "
              << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_scatter_3d_rows(const SsaValue& operand,
                                  const SsaValue& scatter_indices,
                                  const SsaValue& updates) {
        // operand: [B3, B2, R, D], scatter_indices: [B3, B2, N_upd, 3], updates: [B3, B2, N_upd, D]
        SsaValue out = next_val(operand.dtype, operand.dims);
        const std::string scalar_t = format_mlir_tensor_type(operand.dtype, {});
        const std::string old_arg = next_tmp_name("sc_old");
        const std::string new_arg = next_tmp_name("sc_new");
        body_ << "    " << out.name << " = \"stablehlo.scatter\"(" << operand.name << ", "
              << scatter_indices.name << ", " << updates.name << ") ({\n"
              << "    ^bb0(" << old_arg << ": " << scalar_t << ", " << new_arg << ": " << scalar_t << "):\n"
              << "      stablehlo.return " << new_arg << " : " << scalar_t << "\n"
              << "    }) {\n"
              << "      scatter_dimension_numbers = #stablehlo.scatter<\n"
              << "        update_window_dims = [3],\n"
              << "        inserted_window_dims = [0, 1, 2],\n"
              << "        scatter_dims_to_operand_dims = [0, 1, 2],\n"
              << "        index_vector_dim = 3\n"
              << "      >,\n"
              << "      indices_are_sorted = false,\n"
              << "      unique_indices = false\n"
              << "    } : (" << operand.mlir_type() << ", " << scatter_indices.mlir_type() << ", "
              << updates.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_replica_id() {
        SsaValue out = next_val(PjrtDType::U32, {});
        body_ << "    " << out.name << " = \"stablehlo.replica_id\"() : () -> tensor<ui32>\n";
        return out;
    }

    SsaValue emit_dynamic_slice(const SsaValue& x,
                                const std::vector<SsaValue>& start_indices,
                                const std::vector<int64_t>& slice_sizes) {
        SsaValue out = next_val(x.dtype, slice_sizes);
        body_ << "    " << out.name << " = \"stablehlo.dynamic_slice\"(" << x.name;
        for (const auto& idx : start_indices) {
            body_ << ", " << idx.name;
        }
        body_ << ") {\n"
              << "      slice_sizes = array<i64: ";
        for (size_t i = 0; i < slice_sizes.size(); ++i) {
            if (i > 0) body_ << ", ";
            body_ << slice_sizes[i];
        }
        body_ << ">\n    } : (" << x.mlir_type();
        for (const auto& idx : start_indices) {
            body_ << ", " << idx.mlir_type();
        }
        body_ << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_dynamic_update_slice(const SsaValue& operand,
                                       const SsaValue& update,
                                       const std::vector<SsaValue>& start_indices) {
        SsaValue out = next_val(operand.dtype, operand.dims);
        body_ << "    " << out.name << " = \"stablehlo.dynamic_update_slice\"(" << operand.name
              << ", " << update.name;
        for (const auto& idx : start_indices) {
            body_ << ", " << idx.name;
        }
        body_ << ") : (" << operand.mlir_type() << ", " << update.mlir_type();
        for (const auto& idx : start_indices) {
            body_ << ", " << idx.mlir_type();
        }
        body_ << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    static std::string format_replica_groups_1d(int32_t num_chips) {
        std::ostringstream oss = make_classic_stream();
        oss << "dense<[[";
        for (int32_t i = 0; i < num_chips; ++i) {
            if (i > 0) oss << ", ";
            oss << i;
        }
        oss << "]]> : tensor<1x" << num_chips << "xi64>";
        return oss.str();
    }

    SsaValue emit_all_reduce(const SsaValue& x, int32_t num_chips, const char* reducer_op) {
        SsaValue out = next_val(x.dtype, x.dims);
        const std::string scalar_t = format_mlir_tensor_type(x.dtype, {});
        const std::string lhs_arg = next_tmp_name("ar_lhs");
        const std::string rhs_arg = next_tmp_name("ar_rhs");
        const std::string acc_val = next_tmp_name("ar_acc");
        body_ << "    " << out.name << " = \"stablehlo.all_reduce\"(" << x.name << ") ({\n"
              << "    ^bb0(" << lhs_arg << ": " << scalar_t << ", " << rhs_arg << ": " << scalar_t << "):\n"
              << "      " << acc_val << " = " << reducer_op << " " << lhs_arg << ", " << rhs_arg
              << " : " << scalar_t << "\n"
              << "      stablehlo.return " << acc_val << " : " << scalar_t << "\n"
              << "    }) {\n"
              << "      replica_groups = " << format_replica_groups_1d(num_chips) << ",\n"
              << "      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>\n"
              << "    } : (" << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_all_gather(const SsaValue& x, int64_t gather_dim, int32_t num_chips) {
        if (num_chips <= 0) {
            throw std::invalid_argument("emit_all_gather: num_chips must be positive");
        }
        if (gather_dim < 0 || static_cast<size_t>(gather_dim) >= x.dims.size()) {
            throw std::invalid_argument("emit_all_gather: gather_dim out of bounds");
        }
        std::vector<int64_t> out_dims = x.dims;
        out_dims[static_cast<size_t>(gather_dim)] *= static_cast<int64_t>(num_chips);
        SsaValue out = next_val(x.dtype, out_dims);
        body_ << "    " << out.name << " = \"stablehlo.all_gather\"(" << x.name << ") {\n"
              << "      all_gather_dim = " << gather_dim << " : i64,\n"
              << "      replica_groups = " << format_replica_groups_1d(num_chips) << ",\n"
              << "      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>\n"
              << "    } : (" << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    SsaValue emit_reduce_scatter(const SsaValue& x, int64_t scatter_dim, int32_t num_chips, const char* reducer_op) {
        if (num_chips <= 0) {
            throw std::invalid_argument("emit_reduce_scatter: num_chips must be positive");
        }
        if (scatter_dim < 0 || static_cast<size_t>(scatter_dim) >= x.dims.size()) {
            throw std::invalid_argument("emit_reduce_scatter: scatter_dim out of bounds");
        }
        std::vector<int64_t> out_dims = x.dims;
        if (out_dims[static_cast<size_t>(scatter_dim)] % num_chips != 0) {
            throw std::invalid_argument("emit_reduce_scatter: scatter_dim not divisible by num_chips");
        }
        out_dims[static_cast<size_t>(scatter_dim)] /= static_cast<int64_t>(num_chips);
        SsaValue out = next_val(x.dtype, out_dims);
        const std::string scalar_t = format_mlir_tensor_type(x.dtype, {});
        const std::string lhs_arg = next_tmp_name("rs_lhs");
        const std::string rhs_arg = next_tmp_name("rs_rhs");
        const std::string acc_val = next_tmp_name("rs_acc");
        body_ << "    " << out.name << " = \"stablehlo.reduce_scatter\"(" << x.name << ") ({\n"
              << "    ^bb0(" << lhs_arg << ": " << scalar_t << ", " << rhs_arg << ": " << scalar_t << "):\n"
              << "      " << acc_val << " = " << reducer_op << " " << lhs_arg << ", " << rhs_arg
              << " : " << scalar_t << "\n"
              << "      stablehlo.return " << acc_val << " : " << scalar_t << "\n"
              << "    }) {\n"
              << "      scatter_dimension = " << scatter_dim << " : i64,\n"
              << "      replica_groups = " << format_replica_groups_1d(num_chips) << ",\n"
              << "      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>\n"
              << "    } : (" << x.mlir_type() << ") -> " << out.mlir_type() << "\n";
        return out;
    }

    std::string finish(const std::vector<SsaValue>& outputs) const {
        std::ostringstream mod = make_classic_stream();
        mod << "module @ggml_cloudtpu {\n";
        mod << "  func.func public @main(";
        for (size_t i = 0; i < args_.size(); ++i) {
            if (i > 0) mod << ", ";
            mod << args_[i].name << ": " << args_[i].mlir_type();
            if (i < arg_aliases_.size() && arg_aliases_[i] >= 0) {
                mod << " {tf.aliasing_output = " << arg_aliases_[i] << " : i32}";
            }
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
        std::ostringstream oss = make_classic_stream();
        oss << "%v" << next_id_++;
        return {oss.str(), dt, std::move(dims)};
    }

    std::string next_tmp_name(const char* prefix) {
        std::ostringstream oss = make_classic_stream();
        oss << "%" << prefix << "_" << next_id_++;
        return oss.str();
    }

    std::vector<SsaValue> args_;
    std::vector<int32_t> arg_aliases_;
    std::ostringstream body_;
    int next_id_ = 0;
};

// Dequantizes row-major [B3, B2, N_idx, NB, qs_blk] + [B3, B2, N_idx, NB] blocks (used by GET_ROWS
// after gathering only the small selected subset of rows).
SsaValue dequantize_row_major_blocks(StableHloBuilder& b,
                                     TensorStorageKind storage,
                                     const SsaValue& qs,
                                     const SsaValue& d,
                                     const SsaValue* m_opt,
                                     PjrtDType out_dt) {
    const int64_t B3 = d.dims[0];
    const int64_t B2 = d.dims[1];
    const int64_t N = d.dims[2];
    const int64_t NB = d.dims[3];
    const int64_t blk_elems = (storage == TensorStorageKind::Q6_K_SPLIT) ? 16 : 32;
    const int64_t K = NB * blk_elems;
    const std::vector<int64_t> blk_dims = {B3, B2, N, NB, blk_elems};
    const PjrtDType calc_dt = PjrtDType::F32;

    if (storage == TensorStorageKind::Q8_0_SPLIT || storage == TensorStorageKind::Q6_K_SPLIT) {
        SsaValue q_f = b.emit_convert(qs, calc_dt);
        SsaValue d_c = b.emit_convert(d, calc_dt);
        SsaValue d_b = b.emit_broadcast_in_dim(d_c, {0, 1, 2, 3}, blk_dims);
        SsaValue scaled = b.emit_binary("stablehlo.multiply", q_f, d_b);
        return b.emit_convert(b.emit_reshape(scaled, {B3, B2, N, K}), out_dt);
    }
    if (storage == TensorStorageKind::Q4_0_SPLIT || storage == TensorStorageKind::Q4_K_SPLIT) {
        const std::vector<int64_t> half_dims = {B3, B2, N, NB, 16};
        SsaValue mask = b.emit_const_splat_int(0x0F, half_dims, PjrtDType::U8);
        SsaValue shift4 = b.emit_const_splat_int(4, half_dims, PjrtDType::U8);
        SsaValue lo_u8 = b.emit_binary("stablehlo.and", qs, mask);
        SsaValue hi_u8 = b.emit_binary("stablehlo.shift_right_logical", qs, shift4);
        SsaValue cat_i32 = b.emit_convert(b.emit_concatenate({lo_u8, hi_u8}, 4), PjrtDType::S32);
        SsaValue q_f;
        if (storage == TensorStorageKind::Q4_0_SPLIT) {
            SsaValue eight = b.emit_const_splat_int(8, blk_dims, PjrtDType::S32);
            q_f = b.emit_convert(b.emit_binary("stablehlo.subtract", cat_i32, eight), calc_dt);
        } else {
            q_f = b.emit_convert(cat_i32, calc_dt);
        }
        SsaValue d_c = b.emit_convert(d, calc_dt);
        SsaValue d_b = b.emit_broadcast_in_dim(d_c, {0, 1, 2, 3}, blk_dims);
        SsaValue scaled = b.emit_binary("stablehlo.multiply", q_f, d_b);
        if (storage == TensorStorageKind::Q4_K_SPLIT && m_opt != nullptr) {
            SsaValue m_c = b.emit_convert(*m_opt, calc_dt);
            SsaValue m_b = b.emit_broadcast_in_dim(m_c, {0, 1, 2, 3}, blk_dims);
            scaled = b.emit_binary("stablehlo.add", scaled, m_b);
        }
        return b.emit_convert(b.emit_reshape(scaled, {B3, B2, N, K}), out_dt);
    }
    throw std::runtime_error("dequantize_row_major_blocks: unexpected storage kind");
}

// TPU-native K-major N-minor dequantization (zero minor-dim tile padding, zero weight reshape,
// zero broadcast reshape):
//   Q8_0_SPLIT: qs [B3, B2, NB, 32, N] S8,  d [B3, B2, NB, N] F16      -> [B3, B2, NB, 32, N]
//   Q6_K_SPLIT: qs [B3, B2, NB, 16, N] S8,  d [B3, B2, NB, N] F32/BF16 -> [B3, B2, NB, 16, N]
SsaValue dequantize_k_major_blocks(StableHloBuilder& b,
                                   TensorStorageKind storage,
                                   const SsaValue& qs,
                                   const SsaValue& d,
                                   PjrtDType out_dt) {
    const int64_t B3 = d.dims[0];
    const int64_t B2 = d.dims[1];
    const int64_t NB = d.dims[2];
    const int64_t N = d.dims[3];
    const int64_t blk_elems = (storage == TensorStorageKind::Q6_K_SPLIT) ? 16 : 32;
    const std::vector<int64_t> km_dims = {B3, B2, NB, blk_elems, N};

    SsaValue q_f = b.emit_convert(qs, PjrtDType::F32);
    SsaValue d_c = b.emit_convert(d, PjrtDType::F32);
    SsaValue d_b = b.emit_broadcast_in_dim(d_c, {0, 1, 2, 4}, km_dims);
    SsaValue scaled = b.emit_binary("stablehlo.multiply", q_f, d_b);
    return b.emit_convert(scaled, out_dt);
}

// TPU-native K-major N-minor Q4_0/Q4_K dequantization into two [B3, B2, NB, 16, N] halves
// (low nibbles = block[0..15], high nibbles = block[16..31]). Minor dims are (16, N),
// aligning directly with TPU (8, 128) HBM tiles with 0% padding and zero broadcast reshapes.
void dequantize_q4_k_major_halves(StableHloBuilder& b,
                                  TensorStorageKind storage,
                                  const SsaValue& qs,
                                  const SsaValue& d,
                                  const SsaValue* m_opt,
                                  PjrtDType out_dt,
                                  SsaValue& out_lo,
                                  SsaValue& out_hi) {
    const int64_t B3 = d.dims[0];
    const int64_t B2 = d.dims[1];
    const int64_t NB = d.dims[2];
    const int64_t N = d.dims[3];
    const std::vector<int64_t> km_dims = {B3, B2, NB, 16, N};

    SsaValue mask = b.emit_const_splat_int(0x0F, km_dims, PjrtDType::U8);
    SsaValue shift4 = b.emit_const_splat_int(4, km_dims, PjrtDType::U8);
    SsaValue lo_i32 = b.emit_convert(b.emit_binary("stablehlo.and", qs, mask), PjrtDType::S32);
    SsaValue hi_i32 = b.emit_convert(b.emit_binary("stablehlo.shift_right_logical", qs, shift4), PjrtDType::S32);
    if (storage == TensorStorageKind::Q4_0_SPLIT) {
        SsaValue eight = b.emit_const_splat_int(8, km_dims, PjrtDType::S32);
        lo_i32 = b.emit_binary("stablehlo.subtract", lo_i32, eight);
        hi_i32 = b.emit_binary("stablehlo.subtract", hi_i32, eight);
    }
    SsaValue lo_f = b.emit_convert(lo_i32, PjrtDType::F32);
    SsaValue hi_f = b.emit_convert(hi_i32, PjrtDType::F32);

    SsaValue d_c = b.emit_convert(d, PjrtDType::F32);
    SsaValue d_b = b.emit_broadcast_in_dim(d_c, {0, 1, 2, 4}, km_dims);
    SsaValue w_lo = b.emit_binary("stablehlo.multiply", lo_f, d_b);
    SsaValue w_hi = b.emit_binary("stablehlo.multiply", hi_f, d_b);
    if (storage == TensorStorageKind::Q4_K_SPLIT && m_opt != nullptr) {
        SsaValue m_c = b.emit_convert(*m_opt, PjrtDType::F32);
        SsaValue m_b = b.emit_broadcast_in_dim(m_c, {0, 1, 2, 4}, km_dims);
        w_lo = b.emit_binary("stablehlo.add", w_lo, m_b);
        w_hi = b.emit_binary("stablehlo.add", w_hi, m_b);
    }
    out_lo = b.emit_convert(w_lo, out_dt);
    out_hi = b.emit_convert(w_hi, out_dt);
}

// Fallback dense [B3, B2, N, K] materializer for non-MUL_MAT/non-GET_ROWS consumers (e.g. CPY).
SsaValue dequantize_split_weight(StableHloBuilder& b,
                                 TensorStorageKind storage,
                                 const SsaValue& qs,
                                 const SsaValue& d,
                                 const SsaValue* m_opt,
                                 PjrtDType out_dt) {
    const int64_t B3 = d.dims[0];
    const int64_t B2 = d.dims[1];
    const int64_t NB = d.dims[2];
    const int64_t N = d.dims[3];
    const int64_t blk_elems = (storage == TensorStorageKind::Q6_K_SPLIT) ? 16 : 32;
    const int64_t K = NB * blk_elems;

    if (storage == TensorStorageKind::Q8_0_SPLIT || storage == TensorStorageKind::Q6_K_SPLIT) {
        SsaValue km = dequantize_k_major_blocks(b, storage, qs, d, out_dt);
        SsaValue nm = b.emit_transpose(km, {0, 1, 4, 2, 3});
        return b.emit_reshape(nm, {B3, B2, N, K});
    }
    if (storage == TensorStorageKind::Q4_0_SPLIT || storage == TensorStorageKind::Q4_K_SPLIT) {
        SsaValue w_lo, w_hi;
        dequantize_q4_k_major_halves(b, storage, qs, d, m_opt, out_dt, w_lo, w_hi);
        SsaValue lo_nm = b.emit_transpose(w_lo, {0, 1, 4, 2, 3});
        SsaValue hi_nm = b.emit_transpose(w_hi, {0, 1, 4, 2, 3});
        SsaValue cat = b.emit_concatenate({lo_nm, hi_nm}, 4);
        return b.emit_reshape(cat, {B3, B2, N, K});
    }
    throw std::runtime_error("dequantize_split_weight: unexpected storage kind");
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
            throw std::runtime_error("StableHLO broadcast_4d: non-divisible shape src=" +
                                     format_dims_bracket(x.dims) + " dst=" + format_dims_bracket(target_dims));
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

std::string build_canonical_cache_key(const std::string& mlir_text, int32_t spmd_chips) {
    // Structural key combining size + two independent FNV hashes (64-bit normal + 64-bit salted),
    // exact length, and SPMD chip count to prevent collisions without external SHA libraries.
    const uint64_t h1 = fnv1a_64(mlir_text);
    const uint64_t h2 = fnv1a_64(std::string("cloudtpu_v1:") + mlir_text);
    std::ostringstream oss = make_classic_stream();
    oss << std::hex << std::setfill('0') << std::setw(16) << h1
        << std::setw(16) << h2 << "_" << std::dec << mlir_text.size() << "_p" << spmd_chips;
    return oss.str();
}

}  // namespace

EmittedProgram emit_stablehlo_program(const GraphProgramSpec& spec) {
    EmittedProgram prog;
    StableHloBuilder builder;
    const int32_t P = std::max<int32_t>(1, spec.spmd_chips);

    struct TensorHandle {
        TensorStorageKind storage = TensorStorageKind::DENSE;
        TensorShardingKind sharding = TensorShardingKind::REPLICATED;
        int64_t logical_k = 0;
        SsaValue dense;
        SsaValue q_qs;
        SsaValue q_d;
        SsaValue q_m;
        bool has_q_m = false;
    };

    std::unordered_map<uint64_t, TensorHandle> env;

    size_t arg_idx = 0;
    for (const TensorSpec& in : spec.inputs) {
        TensorHandle th;
        th.storage = in.storage;
        th.sharding = (P > 1) ? in.sharding : TensorShardingKind::REPLICATED;
        th.logical_k = in.dims[3];
        const int64_t B3 = in.dims[0];
        const int64_t B2 = in.dims[1];
        int64_t M_or_N = in.dims[2];
        int64_t K = in.dims[3];
        if (th.sharding == TensorShardingKind::SHARD_N) {
            if (M_or_N % P != 0) {
                throw std::runtime_error("emit_stablehlo_program: SHARD_N dim not divisible by spmd_chips");
            }
            M_or_N /= P;
        } else if (th.sharding == TensorShardingKind::SHARD_K) {
            if (K % P != 0) {
                throw std::runtime_error("emit_stablehlo_program: SHARD_K dim not divisible by spmd_chips");
            }
            K /= P;
        }

        if (in.storage == TensorStorageKind::DENSE) {
            std::vector<int64_t> dims = {B3, B2, M_or_N, K};
            const std::string aname = "%arg" + std::to_string(arg_idx++);
            const int32_t alias_out =
                (in.alias_output_idx >= 0 && static_cast<size_t>(in.alias_output_idx) < spec.output_ids.size())
                    ? in.alias_output_idx
                    : -1;
            builder.add_arg(aname, in.dtype, dims, alias_out);
            th.dense = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 0, in.dtype, std::move(dims), alias_out >= 0});
        } else {
            const int64_t blk_elems = (in.storage == TensorStorageKind::Q6_K_SPLIT) ? 16 : 32;
            if (K <= 0 || (K % blk_elems) != 0) {
                throw std::runtime_error("emit_stablehlo_program: quantized tensor K alignment error");
            }
            const int64_t NB = K / blk_elems;
            const PjrtDType q_dt =
                (in.storage == TensorStorageKind::Q8_0_SPLIT || in.storage == TensorStorageKind::Q6_K_SPLIT)
                    ? PjrtDType::S8
                    : PjrtDType::U8;
            const int64_t qs_blk =
                (in.storage == TensorStorageKind::Q8_0_SPLIT || in.storage == TensorStorageKind::Q6_K_SPLIT)
                    ? blk_elems
                    : 16;
            const PjrtDType default_d_dt =
                (in.storage == TensorStorageKind::Q4_K_SPLIT || in.storage == TensorStorageKind::Q6_K_SPLIT)
                    ? PjrtDType::F32
                    : PjrtDType::F16;
            const PjrtDType d_dt = (in.scale_dtype != PjrtDType::INVALID) ? in.scale_dtype : default_d_dt;

            // TPU-native K-major N-minor layout:
            //   qs: [B3, B2, NB, qs_blk, N]  -> minor dims (16/32, N) align with (8, 128) TPU tiles (0% padding)
            //   d:  [B3, B2, NB, N]          -> broadcasts along axis 3 without any reshape
            std::vector<int64_t> qs_dims = {B3, B2, NB, qs_blk, M_or_N};
            const std::string q_name = "%arg" + std::to_string(arg_idx++);
            builder.add_arg(q_name, q_dt, qs_dims);
            th.q_qs = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 0, q_dt, std::move(qs_dims), false});

            const std::vector<int64_t> d_dims = {B3, B2, NB, M_or_N};
            const std::string d_name = "%arg" + std::to_string(arg_idx++);
            builder.add_arg(d_name, d_dt, d_dims);
            th.q_d = builder.arg_val(arg_idx - 1);
            prog.input_bindings.push_back({in.id, 1, d_dt, d_dims, false});

            if (in.storage == TensorStorageKind::Q4_K_SPLIT) {
                const std::string m_name = "%arg" + std::to_string(arg_idx++);
                builder.add_arg(m_name, d_dt, d_dims);
                th.q_m = builder.arg_val(arg_idx - 1);
                th.has_q_m = true;
                prog.input_bindings.push_back({in.id, 2, d_dt, d_dims, false});
            }
        }
        env[in.id] = std::move(th);
    }

    auto resolve_dense = [&](uint64_t id, PjrtDType dequant_dt = PjrtDType::BF16) -> SsaValue {
        const TensorHandle& th = env.at(id);
        if (th.storage == TensorStorageKind::DENSE) {
            return th.dense;
        }
        return dequantize_split_weight(
            builder, th.storage, th.q_qs, th.q_d, th.has_q_m ? &th.q_m : nullptr, dequant_dt);
    };

    auto resolve_replicated = [&](uint64_t id, PjrtDType dequant_dt = PjrtDType::BF16) -> SsaValue {
        const TensorHandle& th = env.at(id);
        SsaValue v = resolve_dense(id, dequant_dt);
        if (P > 1 && th.sharding == TensorShardingKind::SHARD_N) {
            return builder.emit_all_gather(v, 2, P);
        }
        if (P > 1 && th.sharding == TensorShardingKind::SHARD_K) {
            return builder.emit_all_gather(v, 3, P);
        }
        return v;
    };

    for (const OpNodeSpec& op : spec.ops) {
        const std::vector<int64_t> dst_dims = {op.dst.dims[0], op.dst.dims[1], op.dst.dims[2], op.dst.dims[3]};
        SsaValue result;
        TensorShardingKind out_sharding = TensorShardingKind::REPLICATED;

        switch (op.kind) {
            case CloudOpKind::MUL_MAT: {
                const TensorHandle& w_th = env.at(op.src_ids.at(0));
                const TensorShardingKind w_sh = w_th.sharding;
                const TensorShardingKind x_sh = env.at(op.src_ids.at(1)).sharding;
                SsaValue x = (x_sh == TensorShardingKind::SHARD_K && w_sh == TensorShardingKind::SHARD_K)
                                 ? resolve_dense(op.src_ids.at(1), PjrtDType::F32)
                                 : resolve_replicated(op.src_ids.at(1), PjrtDType::F32);
                const bool w_is_f32_dense =
                    (w_th.storage == TensorStorageKind::DENSE && w_th.dense.dtype == PjrtDType::F32);
                const bool use_bf16_mxu = spec.bf16_matmul && !op.high_precision && !w_is_f32_dense;
                const PjrtDType deq_dt =
                    use_bf16_mxu ? PjrtDType::BF16
                                 : ((op.high_precision || x.dtype == PjrtDType::F32) ? PjrtDType::F32 : PjrtDType::BF16);

                // Fast path for ALL K-major N-minor quantized weights (Q8_0, Q6_K, Q4_0, Q4_K):
                // Zero weight reshape, zero transpose, zero broadcast reshape, and 0% (8,128) tile padding!
                if (w_th.storage != TensorStorageKind::DENSE) {
                    const int64_t NB_local = w_th.q_d.dims[2];
                    const int64_t N_local = w_th.q_d.dims[3];
                    const int64_t blk_elems = (w_th.storage == TensorStorageKind::Q6_K_SPLIT) ? 16 : 32;
                    const int64_t K_local = NB_local * blk_elems;

                    if (P > 1 && w_sh == TensorShardingKind::SHARD_K && x_sh != TensorShardingKind::SHARD_K) {
                        SsaValue rid = builder.emit_replica_id();
                        SsaValue rid_i32 = builder.emit_convert(rid, PjrtDType::S32);
                        SsaValue k_local_c = builder.emit_const_splat_int(K_local, {}, PjrtDType::S32);
                        SsaValue k_off = builder.emit_binary("stablehlo.multiply", rid_i32, k_local_c);
                        SsaValue zero_i32 = builder.emit_const_splat_int(0, {}, PjrtDType::S32);
                        x = builder.emit_dynamic_slice(
                            x, {zero_i32, zero_i32, zero_i32, k_off}, {x.dims[0], x.dims[1], x.dims[2], K_local});
                    }

                    const PjrtDType mm_dt =
                        use_bf16_mxu ? PjrtDType::BF16
                                     : ((op.high_precision || deq_dt == PjrtDType::F32 || x.dtype == PjrtDType::F32)
                                            ? PjrtDType::F32
                                            : PjrtDType::BF16);
                    x = builder.emit_convert(x, mm_dt);

                    auto bcast_w5_batch = [&](SsaValue& wt) {
                        // wt: [B3_w, B2_w, D2, D3, D4] -> [B3_x, B2_x, D2, D3, D4]
                        if (wt.dims[0] != x.dims[0] || wt.dims[1] != x.dims[1]) {
                            const int64_t r3 = x.dims[0] / wt.dims[0];
                            const int64_t r2 = x.dims[1] / wt.dims[1];
                            SsaValue wr = builder.emit_reshape(
                                wt, {wt.dims[0], 1, wt.dims[1], 1, wt.dims[2], wt.dims[3], wt.dims[4]});
                            SsaValue wb = builder.emit_broadcast_in_dim(
                                wr, {0, 1, 2, 3, 4, 5, 6},
                                {wt.dims[0], r3, wt.dims[1], r2, wt.dims[2], wt.dims[3], wt.dims[4]});
                            wt = builder.emit_reshape(wb, {x.dims[0], x.dims[1], wt.dims[2], wt.dims[3], wt.dims[4]});
                        }
                    };

                    const std::vector<int64_t> local_out_dims = {x.dims[0], x.dims[1], x.dims[2], N_local};
                    SsaValue local_out;
                    if (w_th.storage == TensorStorageKind::Q8_0_SPLIT ||
                        w_th.storage == TensorStorageKind::Q6_K_SPLIT) {
                        SsaValue w_km = dequantize_k_major_blocks(builder, w_th.storage, w_th.q_qs, w_th.q_d, mm_dt);
                        bcast_w5_batch(w_km);
                        SsaValue x_blk = builder.emit_reshape(x, {x.dims[0], x.dims[1], x.dims[2], NB_local, blk_elems});
                        local_out = builder.emit_dot_general(
                            x_blk, w_km, {0, 1}, {0, 1}, {3, 4}, {2, 3}, local_out_dims, op.dst.dtype,
                            op.high_precision);
                    } else {
                        SsaValue w_lo, w_hi;
                        dequantize_q4_k_major_halves(
                            builder, w_th.storage, w_th.q_qs, w_th.q_d,
                            w_th.has_q_m ? &w_th.q_m : nullptr, mm_dt, w_lo, w_hi);
                        bcast_w5_batch(w_lo);
                        bcast_w5_batch(w_hi);
                        SsaValue x_blk = builder.emit_reshape(x, {x.dims[0], x.dims[1], x.dims[2], NB_local, 2, 16});
                        if (spec.q4_split_dot) {
                            SsaValue x_lo = builder.emit_reshape(
                                builder.emit_slice(x_blk, {0, 0, 0, 0, 0, 0},
                                                   {x.dims[0], x.dims[1], x.dims[2], NB_local, 1, 16}),
                                {x.dims[0], x.dims[1], x.dims[2], NB_local, 16});
                            SsaValue x_hi = builder.emit_reshape(
                                builder.emit_slice(x_blk, {0, 0, 0, 0, 1, 0},
                                                   {x.dims[0], x.dims[1], x.dims[2], NB_local, 2, 16}),
                                {x.dims[0], x.dims[1], x.dims[2], NB_local, 16});
                            SsaValue p_lo = builder.emit_dot_general(
                                x_lo, w_lo, {0, 1}, {0, 1}, {3, 4}, {2, 3}, local_out_dims, op.dst.dtype,
                                op.high_precision);
                            SsaValue p_hi = builder.emit_dot_general(
                                x_hi, w_hi, {0, 1}, {0, 1}, {3, 4}, {2, 3}, local_out_dims, op.dst.dtype,
                                op.high_precision);
                            local_out = builder.emit_binary("stablehlo.add", p_lo, p_hi);
                        } else {
                            // Single dot_general reading packed U8 nibbles once; minor dims remain (16, N_local).
                            SsaValue w_lo6 = builder.emit_reshape(
                                w_lo, {x.dims[0], x.dims[1], NB_local, 1, 16, N_local});
                            SsaValue w_hi6 = builder.emit_reshape(
                                w_hi, {x.dims[0], x.dims[1], NB_local, 1, 16, N_local});
                            SsaValue w_cat = builder.emit_concatenate({w_lo6, w_hi6}, 3);
                            local_out = builder.emit_dot_general(
                                x_blk, w_cat, {0, 1}, {0, 1}, {3, 4, 5}, {2, 3, 4}, local_out_dims, op.dst.dtype,
                                op.high_precision);
                        }
                    }

                    if (P > 1 && w_sh == TensorShardingKind::SHARD_N) {
                        if (op.keep_sharded_out) {
                            result = local_out;
                            out_sharding = TensorShardingKind::SHARD_K;
                        } else {
                            result = builder.emit_all_gather(local_out, 3, P);
                            out_sharding = TensorShardingKind::REPLICATED;
                        }
                    } else if (P > 1 && w_sh == TensorShardingKind::SHARD_K) {
                        result = builder.emit_all_reduce(local_out, P, "stablehlo.add");
                        out_sharding = TensorShardingKind::REPLICATED;
                    } else {
                        result = local_out;
                    }
                    break;
                }

                SsaValue w = resolve_dense(op.src_ids.at(0), deq_dt);

                if (op.high_precision) {
                    w = builder.emit_convert(w, PjrtDType::F32);
                    x = builder.emit_convert(x, PjrtDType::F32);
                } else if (use_bf16_mxu) {
                    w = builder.emit_convert(w, PjrtDType::BF16);
                    x = builder.emit_convert(x, PjrtDType::BF16);
                } else if (w.dtype != x.dtype) {
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

                if (P > 1 && w_sh == TensorShardingKind::SHARD_N) {
                    const std::vector<int64_t> local_out_dims = {x.dims[0], x.dims[1], x.dims[2], w.dims[2]};
                    SsaValue local_out = builder.emit_dot_general(
                        x, w, {0, 1}, {0, 1}, {3}, {3}, local_out_dims, op.dst.dtype, op.high_precision);
                    if (op.keep_sharded_out) {
                        result = local_out;
                        out_sharding = TensorShardingKind::SHARD_K;
                    } else {
                        result = builder.emit_all_gather(local_out, 3, P);
                        out_sharding = TensorShardingKind::REPLICATED;
                    }
                } else if (P > 1 && w_sh == TensorShardingKind::SHARD_K) {
                    if (x_sh != TensorShardingKind::SHARD_K) {
                        const int64_t K_local = w.dims[3];
                        SsaValue rid = builder.emit_replica_id();
                        SsaValue rid_i32 = builder.emit_convert(rid, PjrtDType::S32);
                        SsaValue k_local_c = builder.emit_const_splat_int(K_local, {}, PjrtDType::S32);
                        SsaValue k_off = builder.emit_binary("stablehlo.multiply", rid_i32, k_local_c);
                        SsaValue zero_i32 = builder.emit_const_splat_int(0, {}, PjrtDType::S32);
                        x = builder.emit_dynamic_slice(
                            x, {zero_i32, zero_i32, zero_i32, k_off}, {x.dims[0], x.dims[1], x.dims[2], K_local});
                    }
                    SsaValue partial = builder.emit_dot_general(
                        x, w, {0, 1}, {0, 1}, {3}, {3}, dst_dims, op.dst.dtype, op.high_precision);
                    result = builder.emit_all_reduce(partial, P, "stablehlo.add");
                    out_sharding = TensorShardingKind::REPLICATED;
                } else {
                    result = builder.emit_dot_general(
                        x, w, {0, 1}, {0, 1}, {3}, {3}, dst_dims, op.dst.dtype, op.high_precision);
                }
                break;
            }
            case CloudOpKind::RMS_NORM: {
                SsaValue x_in = resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
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
                SsaValue x_in = resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
                SsaValue pos_in = resolve_replicated(op.src_ids.at(1), PjrtDType::S32);
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
                    SsaValue ff_raw = resolve_replicated(op.src_ids[2], PjrtDType::F32);
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
                const TensorShardingKind x_sh = env.at(op.src_ids.at(0)).sharding;
                SsaValue x = (x_sh == TensorShardingKind::SHARD_K)
                                 ? resolve_dense(op.src_ids.at(0), PjrtDType::F32)
                                 : resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
                result = builder.emit_convert(emit_silu(builder, x), op.dst.dtype);
                out_sharding = (x_sh == TensorShardingKind::SHARD_K) ? TensorShardingKind::SHARD_K
                                                                     : TensorShardingKind::REPLICATED;
                break;
            }
            case CloudOpKind::SWIGLU: {
                SsaValue gate, up;
                if (op.src_ids.size() >= 2 && op.src_ids[1] != 0) {
                    const TensorShardingKind g_sh = env.at(op.src_ids[0]).sharding;
                    const TensorShardingKind u_sh = env.at(op.src_ids[1]).sharding;
                    if (g_sh == TensorShardingKind::SHARD_K && u_sh == TensorShardingKind::SHARD_K) {
                        gate = resolve_dense(op.src_ids[0], PjrtDType::F32);
                        up = resolve_dense(op.src_ids[1], PjrtDType::F32);
                        out_sharding = TensorShardingKind::SHARD_K;
                    } else {
                        gate = resolve_replicated(op.src_ids[0], PjrtDType::F32);
                        up = resolve_replicated(op.src_ids[1], PjrtDType::F32);
                    }
                } else {
                    SsaValue packed = resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
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
                SsaValue x_in = resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
                SsaValue s = builder.emit_convert(x_in, PjrtDType::F32);
                if (op.soft_max.scale != 1.0f) {
                    SsaValue sc = builder.emit_const_splat_f32(op.soft_max.scale, s.dims, PjrtDType::F32);
                    s = builder.emit_binary("stablehlo.multiply", s, sc);
                }
                if (op.src_ids.size() > 1 && op.src_ids[1] != 0) {
                    SsaValue m_raw = resolve_replicated(op.src_ids[1], PjrtDType::F32);
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
                SsaValue one_f = builder.emit_const_splat_f32(1.0f, s.dims, PjrtDType::F32);
                SsaValue safe_rsum = builder.emit_binary("stablehlo.maximum", rsum_b, one_f);
                SsaValue probs = builder.emit_binary("stablehlo.divide", exps, safe_rsum);
                result = builder.emit_convert(probs, op.dst.dtype);
                break;
            }
            case CloudOpKind::GET_ROWS: {
                const TensorHandle& tbl = env.at(op.src_ids.at(0));
                SsaValue idx_raw = resolve_replicated(op.src_ids.at(1), PjrtDType::S32);
                // idx_raw: [1, B3, B2, N_idx] -> [B3, B2, N_idx, 1]
                const int64_t B3 = dst_dims[0];
                const int64_t B2 = dst_dims[1];
                const int64_t N_idx = dst_dims[2];
                const std::vector<int64_t> idx_col_dims = {B3, B2, N_idx, 1};
                SsaValue idx3 = builder.emit_iota(0, idx_col_dims, PjrtDType::S32);
                SsaValue idx2 = builder.emit_iota(1, idx_col_dims, PjrtDType::S32);
                SsaValue idx1 = builder.emit_reshape(idx_raw, idx_col_dims);

                const bool is_shard_n = (P > 1 && tbl.sharding == TensorShardingKind::SHARD_N);
                SsaValue start_indices;
                SsaValue in_range;
                if (is_shard_n) {
                    // Row-sharded lookup: gather only the N_idx requested rows on each replica's
                    // local shard, mask out non-local rows to 0, and all_reduce(add) the small
                    // [B3, B2, N_idx, K] result instead of all-gathering the entire table.
                    // K-major quantized tables store N along dim 3 ([B3, B2, NB, N_local]).
                    const int64_t local_rows =
                        (tbl.storage == TensorStorageKind::DENSE) ? tbl.dense.dims[2] : tbl.q_d.dims[3];
                    SsaValue rid_s32 = builder.emit_convert(builder.emit_replica_id(), PjrtDType::S32);
                    SsaValue rid_b = builder.emit_broadcast_in_dim(rid_s32, {}, idx_col_dims);
                    SsaValue local_rows_c = builder.emit_const_splat_int(local_rows, idx_col_dims, PjrtDType::S32);
                    SsaValue row_start = builder.emit_binary("stablehlo.multiply", rid_b, local_rows_c);
                    SsaValue row_end = builder.emit_binary("stablehlo.add", row_start, local_rows_c);
                    SsaValue ge_start = builder.emit_compare("GE", idx1, row_start);
                    SsaValue lt_end = builder.emit_compare("LT", idx1, row_end);
                    in_range = builder.emit_binary("stablehlo.and", ge_start, lt_end);

                    SsaValue rel_idx = builder.emit_binary("stablehlo.subtract", idx1, row_start);
                    SsaValue zero_idx = builder.emit_const_splat_int(0, idx_col_dims, PjrtDType::S32);
                    SsaValue max_idx = builder.emit_const_splat_int(
                        std::max<int64_t>(0, local_rows - 1), idx_col_dims, PjrtDType::S32);
                    SsaValue clamped_lo = builder.emit_binary("stablehlo.maximum", rel_idx, zero_idx);
                    SsaValue local_idx1 = builder.emit_binary("stablehlo.minimum", clamped_lo, max_idx);
                    start_indices = builder.emit_concatenate({idx3, idx2, local_idx1}, 3);
                } else {
                    start_indices = builder.emit_concatenate({idx3, idx2, idx1}, 3);
                }

                SsaValue local_rows_val;
                if (tbl.storage == TensorStorageKind::DENSE) {
                    const std::vector<int64_t> local_dst = {B3, B2, N_idx, tbl.dense.dims[3]};
                    SsaValue gathered = builder.emit_gather_3d_rows(tbl.dense, start_indices, local_dst);
                    local_rows_val = builder.emit_convert(gathered, op.dst.dtype);
                } else {
                    // Gather only the selected N_idx rows along the last axis (N) from K-major tables,
                    // producing row-major [B3, B2, N_idx, NB, qs_blk] + [B3, B2, N_idx, NB], then dequantize!
                    const int64_t NB = tbl.q_d.dims[2];
                    const int64_t qs_blk = tbl.q_qs.dims[3];
                    SsaValue g_qs = builder.emit_gather_3d_last_axis(
                        tbl.q_qs, start_indices, {B3, B2, N_idx, NB, qs_blk});
                    SsaValue g_d = builder.emit_gather_3d_last_axis(
                        tbl.q_d, start_indices, {B3, B2, N_idx, NB});
                    SsaValue g_m;
                    const SsaValue* g_m_ptr = nullptr;
                    if (tbl.has_q_m) {
                        g_m = builder.emit_gather_3d_last_axis(
                            tbl.q_m, start_indices, {B3, B2, N_idx, NB});
                        g_m_ptr = &g_m;
                    }
                    local_rows_val = dequantize_row_major_blocks(
                        builder, tbl.storage, g_qs, g_d, g_m_ptr, op.dst.dtype);
                }

                if (is_shard_n) {
                    SsaValue mask_b = builder.emit_broadcast_in_dim(in_range, {0, 1, 2, 3}, dst_dims);
                    const bool dst_is_int = (op.dst.dtype == PjrtDType::S8 || op.dst.dtype == PjrtDType::U8 ||
                                             op.dst.dtype == PjrtDType::S16 || op.dst.dtype == PjrtDType::S32 ||
                                             op.dst.dtype == PjrtDType::U32 || op.dst.dtype == PjrtDType::S64);
                    SsaValue zeros_dst = dst_is_int
                                             ? builder.emit_const_splat_int(0, dst_dims, op.dst.dtype)
                                             : builder.emit_const_splat_f32(0.0f, dst_dims, op.dst.dtype);
                    SsaValue masked = builder.emit_select(mask_b, local_rows_val, zeros_dst);
                    result = builder.emit_all_reduce(masked, P, "stablehlo.add");
                } else if (P > 1 && tbl.sharding == TensorShardingKind::SHARD_K) {
                    result = builder.emit_all_gather(local_rows_val, 3, P);
                } else {
                    result = local_rows_val;
                }
                break;
            }
            case CloudOpKind::SET_ROWS: {
                // src_ids: [0]=src_updates (b), [1]=row_indices (c), [2]=dst_table (a)
                SsaValue upd_raw = resolve_replicated(op.src_ids.at(0), op.dst.dtype);
                SsaValue idx_raw = resolve_replicated(op.src_ids.at(1), PjrtDType::S32);
                SsaValue tbl = resolve_replicated(op.src_ids.at(2), op.dst.dtype);
                SsaValue upd = builder.emit_convert(upd_raw, tbl.dtype);
                const int64_t B3 = upd.dims[0];
                const int64_t B2 = upd.dims[1];
                const int64_t N_upd = upd.dims[2];
                // Preserve S64 indices to avoid truncating valid >INT32_MAX row indices.
                const PjrtDType idx_dt =
                    (idx_raw.dtype == PjrtDType::S64) ? PjrtDType::S64 : PjrtDType::S32;
                // Single-token decode fast path: emit O(1) in-place dynamic_update_slice instead of
                // scatter when updating a single contiguous row slice across all batches/heads.
                // Guard out-of-range indices (< 0 or >= n_rows) by selecting the existing row at
                // the clamped destination before the update, matching scatter's discard semantics
                // while keeping dynamic_update_slice as the sole final user of tbl for in-place donation.
                if (N_upd == 1 && idx_raw.dims[0] == 1 && idx_raw.dims[1] == 1 &&
                    idx_raw.dims[2] == 1 && idx_raw.dims[3] == 1 &&
                    upd.dims[0] == tbl.dims[0] && upd.dims[1] == tbl.dims[1] && upd.dims[3] == tbl.dims[3]) {
                    SsaValue idx_typed = builder.emit_convert(idx_raw, idx_dt);
                    SsaValue row_scalar = builder.emit_reshape(idx_typed, {});
                    SsaValue zero_idx = builder.emit_const_splat_int(0, {}, idx_dt);
                    SsaValue n_rows_c = builder.emit_const_splat_int(tbl.dims[2], {}, idx_dt);
                    SsaValue ge_zero = builder.emit_compare("GE", row_scalar, zero_idx);
                    SsaValue lt_rows = builder.emit_compare("LT", row_scalar, n_rows_c);
                    SsaValue in_bounds = builder.emit_binary("stablehlo.and", ge_zero, lt_rows);
                    SsaValue in_bounds_b = builder.emit_broadcast_in_dim(in_bounds, {}, upd.dims);
                    SsaValue cur_row = builder.emit_dynamic_slice(
                        tbl, {zero_idx, zero_idx, row_scalar, zero_idx}, upd.dims);
                    SsaValue safe_upd = builder.emit_select(in_bounds_b, upd, cur_row);
                    SsaValue updated = builder.emit_dynamic_update_slice(
                        tbl, safe_upd, {zero_idx, zero_idx, row_scalar, zero_idx});
                    result = builder.emit_convert(updated, op.dst.dtype);
                    break;
                }
                // idx_raw GGML [N_upd, B2_i, B3_i, 1] -> HLO [1, B3_i, B2_i, N_upd] -> reshape [B3_i, B2_i, N_upd, 1]
                SsaValue idx_typed = builder.emit_convert(idx_raw, idx_dt);
                SsaValue idx_4d = builder.emit_reshape(idx_typed, {idx_raw.dims[1], idx_raw.dims[2], idx_raw.dims[3], 1});
                const std::vector<int64_t> idx_col_dims = {B3, B2, N_upd, 1};
                SsaValue idx_b = broadcast_4d(builder, idx_4d, idx_col_dims);
                SsaValue idx3 = builder.emit_iota(0, idx_col_dims, idx_dt);
                SsaValue idx2 = builder.emit_iota(1, idx_col_dims, idx_dt);
                SsaValue scatter_indices = builder.emit_concatenate({idx3, idx2, idx_b}, 3);
                SsaValue scattered = builder.emit_scatter_3d_rows(tbl, scatter_indices, upd);
                result = builder.emit_convert(scattered, op.dst.dtype);
                break;
            }
            case CloudOpKind::ADD:
            case CloudOpKind::MUL: {
                const TensorHandle& a_th = env.at(op.src_ids.at(0));
                const TensorHandle& b_th = env.at(op.src_ids.at(1));
                SsaValue a, b;
                std::vector<int64_t> calc_dims = dst_dims;
                // Only run the local SHARD_K fast path when both operands have equal logical K
                // widths matching the output; unequal K widths require periodic repeat-broadcast
                // over the full logical K axis, so gather via resolve_replicated first.
                if (P > 1 &&
                    a_th.sharding == TensorShardingKind::SHARD_K &&
                    b_th.sharding == TensorShardingKind::SHARD_K &&
                    a_th.logical_k == b_th.logical_k &&
                    a_th.logical_k == dst_dims[3]) {
                    a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                    b = resolve_dense(op.src_ids.at(1), op.dst.dtype);
                    calc_dims[3] /= P;
                    out_sharding = TensorShardingKind::SHARD_K;
                } else {
                    a = resolve_replicated(op.src_ids.at(0), op.dst.dtype);
                    b = resolve_replicated(op.src_ids.at(1), op.dst.dtype);
                }
                SsaValue ab = broadcast_4d(builder, a, calc_dims);
                SsaValue bb = broadcast_4d(builder, b, calc_dims);
                const PjrtDType calc_dt = (op.dst.dtype == PjrtDType::F32) ? PjrtDType::F32 : PjrtDType::F32;
                SsaValue af = builder.emit_convert(ab, calc_dt);
                SsaValue bf = builder.emit_convert(bb, calc_dt);
                const char* mop = (op.kind == CloudOpKind::ADD) ? "stablehlo.add" : "stablehlo.multiply";
                SsaValue res_f = builder.emit_binary(mop, af, bf);
                result = builder.emit_convert(res_f, op.dst.dtype);
                break;
            }
            case CloudOpKind::SCALE: {
                const TensorShardingKind a_sh = env.at(op.src_ids.at(0)).sharding;
                SsaValue a = (a_sh == TensorShardingKind::SHARD_K)
                                 ? resolve_dense(op.src_ids.at(0), PjrtDType::F32)
                                 : resolve_replicated(op.src_ids.at(0), PjrtDType::F32);
                SsaValue af = builder.emit_convert(a, PjrtDType::F32);
                SsaValue sc = builder.emit_const_splat_f32(op.scale, af.dims, PjrtDType::F32);
                SsaValue res_f = builder.emit_binary("stablehlo.multiply", af, sc);
                if (op.bias != 0.0f) {
                    SsaValue bi = builder.emit_const_splat_f32(op.bias, af.dims, PjrtDType::F32);
                    res_f = builder.emit_binary("stablehlo.add", res_f, bi);
                }
                result = builder.emit_convert(res_f, op.dst.dtype);
                out_sharding = (a_sh == TensorShardingKind::SHARD_K) ? TensorShardingKind::SHARD_K
                                                                     : TensorShardingKind::REPLICATED;
                break;
            }
            case CloudOpKind::CPY:
            case CloudOpKind::RESHAPE: {
                SsaValue a = resolve_replicated(op.src_ids.at(0), op.dst.dtype);
                SsaValue c = builder.emit_convert(a, op.dst.dtype);
                result = builder.emit_reshape(c, dst_dims);
                break;
            }
            case CloudOpKind::TRANSPOSE: {
                SsaValue a = resolve_replicated(op.src_ids.at(0), op.dst.dtype);
                SsaValue t = builder.emit_transpose(a, {op.perm[0], op.perm[1], op.perm[2], op.perm[3]});
                result = builder.emit_convert(t, op.dst.dtype);
                break;
            }
            case CloudOpKind::SLICE: {
                SsaValue a = resolve_replicated(op.src_ids.at(0), op.dst.dtype);
                SsaValue cur = op.slice_reshape.empty() ? a : builder.emit_reshape(a, op.slice_reshape);
                SsaValue sl = builder.emit_slice(cur, op.slice_starts, op.slice_limits);
                SsaValue rs = builder.emit_reshape(sl, dst_dims);
                result = builder.emit_convert(rs, op.dst.dtype);
                break;
            }
            case CloudOpKind::ALL_REDUCE_SUM: {
                SsaValue a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                SsaValue c = builder.emit_convert(a, op.dst.dtype);
                result = builder.emit_all_reduce(c, P, "stablehlo.add");
                break;
            }
            case CloudOpKind::ALL_GATHER: {
                SsaValue a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                SsaValue c = builder.emit_convert(a, op.dst.dtype);
                result = builder.emit_all_gather(c, op.collective_dim, P);
                break;
            }
            case CloudOpKind::REDUCE_SCATTER_SUM: {
                SsaValue a = resolve_dense(op.src_ids.at(0), op.dst.dtype);
                SsaValue c = builder.emit_convert(a, op.dst.dtype);
                result = builder.emit_reduce_scatter(c, op.collective_dim, P, "stablehlo.add");
                // Keep explicit reduce_scatter outputs replica-local so downstream CPY/RESHAPE
                // operates on the scattered shape rather than all-gathering into a mismatched shape.
                out_sharding = TensorShardingKind::REPLICATED;
                break;
            }
            case CloudOpKind::FLASH_ATTN_EXT: {
                // q: [B, H_Q, N_Q, DK], k: [B, H_KV, N_KV, DK], v: [B, H_KV, N_KV, DV], mask (opt): [B_m, H_m, N_Q_pad, N_KV]
                // dst: [B, N_Q, H_Q, DV] F32
                SsaValue q = builder.emit_convert(resolve_replicated(op.src_ids.at(0), PjrtDType::F32), PjrtDType::F32);
                SsaValue k = builder.emit_convert(resolve_replicated(op.src_ids.at(1), PjrtDType::F32), PjrtDType::F32);
                SsaValue v = builder.emit_convert(resolve_replicated(op.src_ids.at(2), PjrtDType::F32), PjrtDType::F32);

                const int64_t B = q.dims[0];
                const int64_t H_Q = q.dims[1];
                const int64_t N_Q = q.dims[2];
                const int64_t H_KV = k.dims[1];
                const int64_t N_KV = k.dims[2];
                const int64_t DK = k.dims[3];
                const int64_t DV = v.dims[3];

                if (H_KV != H_Q) {
                    const int64_t r = H_Q / H_KV;
                    auto expand_kv = [&](const SsaValue& t, int64_t d_last) -> SsaValue {
                        SsaValue tr = builder.emit_reshape(t, {B, H_KV, 1, N_KV, d_last});
                        SsaValue tb = builder.emit_broadcast_in_dim(tr, {0, 1, 2, 3, 4}, {B, H_KV, r, N_KV, d_last});
                        return builder.emit_reshape(tb, {B, H_Q, N_KV, d_last});
                    };
                    k = expand_kv(k, DK);
                    v = expand_kv(v, DV);
                }

                // scores = q @ k^T : [B, H_Q, N_Q, N_KV]
                const std::vector<int64_t> sc_dims = {B, H_Q, N_Q, N_KV};
                const bool has_mask = (op.src_ids.size() > 3 && op.src_ids[3] != 0);
                SsaValue m_b;
                SsaValue visible;
                SsaValue zeros_sc;
                SsaValue neg_inf_sc;
                SsaValue zero_scalar = builder.emit_const_splat_f32(0.0f, {}, PjrtDType::F32);

                if (has_mask) {
                    SsaValue m = builder.emit_convert(resolve_replicated(op.src_ids[3], PjrtDType::F32), PjrtDType::F32);
                    if (m.dims[2] > N_Q) {
                        m = builder.emit_slice(m, {0, 0, 0, 0}, {m.dims[0], m.dims[1], N_Q, m.dims[3]});
                    }
                    m_b = broadcast_4d(builder, m, sc_dims);
                    neg_inf_sc = builder.emit_const_splat_f32(
                        -std::numeric_limits<float>::infinity(), sc_dims, PjrtDType::F32);
                    visible = builder.emit_compare("GT", m_b, neg_inf_sc);
                    zeros_sc = builder.emit_const_splat_f32(0.0f, sc_dims, PjrtDType::F32);
                    SsaValue ones_sc = builder.emit_const_splat_f32(1.0f, sc_dims, PjrtDType::F32);
                    SsaValue vis_f32 = builder.emit_select(visible, ones_sc, zeros_sc);
                    SsaValue any_vis_f32 = builder.emit_reduce_axis(vis_f32, zero_scalar, 2, "stablehlo.maximum");
                    SsaValue any_vis = builder.emit_compare(
                        "GT", any_vis_f32, builder.emit_const_splat_f32(0.0f, {B, H_Q, N_KV}, PjrtDType::F32));
                    // Zero out K and V rows that are masked for every query before matmuls so
                    // uninitialized/other-sequence non-finite KV entries cannot contaminate results.
                    SsaValue any_vis_k = builder.emit_broadcast_in_dim(any_vis, {0, 1, 2}, {B, H_Q, N_KV, DK});
                    k = builder.emit_select(
                        any_vis_k, k, builder.emit_const_splat_f32(0.0f, {B, H_Q, N_KV, DK}, PjrtDType::F32));
                    SsaValue any_vis_v = builder.emit_broadcast_in_dim(any_vis, {0, 1, 2}, {B, H_Q, N_KV, DV});
                    v = builder.emit_select(
                        any_vis_v, v, builder.emit_const_splat_f32(0.0f, {B, H_Q, N_KV, DV}, PjrtDType::F32));
                }

                const bool use_bf16_attn = spec.bf16_flash_attn && !op.high_precision;
                SsaValue q_mm = use_bf16_attn ? builder.emit_convert(q, PjrtDType::BF16) : q;
                SsaValue k_mm = use_bf16_attn ? builder.emit_convert(k, PjrtDType::BF16) : k;
                SsaValue scores = builder.emit_dot_general(
                    q_mm, k_mm, {0, 1}, {0, 1}, {3}, {3}, sc_dims, PjrtDType::F32, op.high_precision);
                if (op.flash_attn.scale != 1.0f) {
                    SsaValue sc = builder.emit_const_splat_f32(op.flash_attn.scale, sc_dims, PjrtDType::F32);
                    scores = builder.emit_binary("stablehlo.multiply", scores, sc);
                }
                if (op.flash_attn.logit_softcap != 0.0f) {
                    const float cap = op.flash_attn.logit_softcap;
                    SsaValue inv_cap = builder.emit_const_splat_f32(1.0f / cap, sc_dims, PjrtDType::F32);
                    SsaValue cap_c = builder.emit_const_splat_f32(cap, sc_dims, PjrtDType::F32);
                    scores = builder.emit_binary("stablehlo.multiply", scores, inv_cap);
                    scores = builder.emit_unary("stablehlo.tanh", scores);
                    scores = builder.emit_binary("stablehlo.multiply", scores, cap_c);
                }
                if (has_mask) {
                    if (op.flash_attn.max_bias > 0.0f) {
                        const uint32_t n_head = static_cast<uint32_t>(H_Q);
                        const uint32_t n_head_log2 = 1U << static_cast<uint32_t>(std::floor(std::log2(n_head)));
                        const float m0 = std::pow(2.0f, -op.flash_attn.max_bias / static_cast<float>(n_head_log2));
                        const float m1 = std::pow(2.0f, -(op.flash_attn.max_bias * 0.5f) / static_cast<float>(n_head_log2));
                        std::vector<float> slopes(n_head);
                        for (uint32_t h = 0; h < n_head; ++h) {
                            slopes[h] = (h < n_head_log2)
                                            ? std::pow(m0, static_cast<float>(h + 1))
                                            : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
                        }
                        SsaValue sl_c = builder.emit_const_1d_f32(slopes, {1, H_Q, 1, 1});
                        SsaValue sl_b = builder.emit_broadcast_in_dim(sl_c, {0, 1, 2, 3}, sc_dims);
                        m_b = builder.emit_binary("stablehlo.multiply", m_b, sl_b);
                    }
                    SsaValue scores_added = builder.emit_binary("stablehlo.add", scores, m_b);
                    scores = builder.emit_select(visible, scores_added, neg_inf_sc);
                }

                // Softmax along N_KV (dim 3)
                SsaValue neg_flt = builder.emit_const_splat_f32(-3.40282347e+38f, {}, PjrtDType::F32);
                SsaValue rmax = builder.emit_reduce_axis(scores, neg_flt, 3, "stablehlo.maximum");
                SsaValue rmax_b = builder.emit_broadcast_in_dim(rmax, {0, 1, 2}, sc_dims);
                SsaValue shifted = builder.emit_binary("stablehlo.subtract", scores, rmax_b);
                SsaValue exps = builder.emit_unary("stablehlo.exponential", shifted);
                if (has_mask) {
                    exps = builder.emit_select(visible, exps, zeros_sc);
                }
                SsaValue rsum = builder.emit_reduce_axis(exps, zero_scalar, 3, "stablehlo.add");
                SsaValue rsum_b = builder.emit_broadcast_in_dim(rsum, {0, 1, 2}, sc_dims);
                SsaValue one_f = builder.emit_const_splat_f32(1.0f, sc_dims, PjrtDType::F32);
                SsaValue safe_rsum = builder.emit_binary("stablehlo.maximum", rsum_b, one_f);
                SsaValue probs = builder.emit_binary("stablehlo.divide", exps, safe_rsum);

                // out_bhqd = probs @ v : [B, H_Q, N_Q, N_KV] x [B, H_Q, N_KV, DV] -> [B, H_Q, N_Q, DV]
                SsaValue probs_mm = use_bf16_attn ? builder.emit_convert(probs, PjrtDType::BF16) : probs;
                SsaValue out_bhqd;
                if (has_mask) {
                    // Prevent 0.0 * (NaN/Inf) contamination when one query masks a KV row that
                    // another query in the same batch attends to, while still propagating NaN to
                    // any query whose visible attended value is non-finite.
                    const std::vector<int64_t> v_dims = {B, H_Q, N_KV, DV};
                    SsaValue v_zeros = builder.emit_const_splat_f32(0.0f, v_dims, PjrtDType::F32);
                    SsaValue v_ones = builder.emit_const_splat_f32(1.0f, v_dims, PjrtDType::F32);
                    SsaValue v_diff = builder.emit_binary("stablehlo.subtract", v, v);
                    SsaValue v_is_fin = builder.emit_compare("EQ", v_diff, v_zeros);
                    SsaValue v_fin = builder.emit_select(v_is_fin, v, v_zeros);
                    SsaValue v_fin_mm = use_bf16_attn ? builder.emit_convert(v_fin, PjrtDType::BF16) : v_fin;
                    out_bhqd = builder.emit_dot_general(
                        probs_mm, v_fin_mm, {0, 1}, {0, 1}, {3}, {2}, {B, H_Q, N_Q, DV}, PjrtDType::F32, op.high_precision);
                    SsaValue v_bad = builder.emit_select(v_is_fin, v_zeros, v_ones);
                    SsaValue q_bad = builder.emit_dot_general(
                        probs, v_bad, {0, 1}, {0, 1}, {3}, {2}, {B, H_Q, N_Q, DV}, PjrtDType::F32, false);
                    SsaValue q_has_bad = builder.emit_compare(
                        "GT", q_bad, builder.emit_const_splat_f32(0.0f, {B, H_Q, N_Q, DV}, PjrtDType::F32));
                    SsaValue nan_out = builder.emit_const_splat_f32(
                        std::numeric_limits<float>::quiet_NaN(), {B, H_Q, N_Q, DV}, PjrtDType::F32);
                    out_bhqd = builder.emit_select(q_has_bad, nan_out, out_bhqd);
                } else {
                    SsaValue v_mm = use_bf16_attn ? builder.emit_convert(v, PjrtDType::BF16) : v;
                    out_bhqd = builder.emit_dot_general(
                        probs_mm, v_mm, {0, 1}, {0, 1}, {3}, {2}, {B, H_Q, N_Q, DV}, PjrtDType::F32, op.high_precision);
                }
                // Permute [B, H_Q, N_Q, DV] -> [B, N_Q, H_Q, DV]
                SsaValue out_permuted = builder.emit_transpose(out_bhqd, {0, 2, 1, 3});
                result = builder.emit_convert(out_permuted, op.dst.dtype);
                break;
            }
        }

        TensorHandle out_th;
        out_th.storage = TensorStorageKind::DENSE;
        out_th.sharding = out_sharding;
        out_th.logical_k = op.dst.dims[3];
        out_th.dense = std::move(result);
        env[op.dst.id] = std::move(out_th);
    }

    std::vector<SsaValue> return_vals;
    return_vals.reserve(spec.output_ids.size());
    for (uint64_t oid : spec.output_ids) {
        const TensorHandle& th = env.at(oid);
        SsaValue ov = th.dense;
        if (P > 1) {
            if (th.sharding == TensorShardingKind::SHARD_N) {
                ov = builder.emit_all_gather(ov, 2, P);
            } else if (th.sharding == TensorShardingKind::SHARD_K) {
                ov = builder.emit_all_gather(ov, 3, P);
            }
        }
        return_vals.push_back(ov);
        prog.output_bindings.push_back({oid, ov.dtype, ov.dims});
    }

    prog.mlir_text = builder.finish(return_vals);
    prog.cache_key = build_canonical_cache_key(prog.mlir_text, P);
    return prog;
}

}  // namespace cloudtpu
