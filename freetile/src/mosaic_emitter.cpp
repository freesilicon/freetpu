#include "freetile/mosaic_emitter.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace freetile {

namespace {

std::string format_memref_vmem(const std::vector<int64_t>& shape, DType dt) {
    std::ostringstream oss;
    oss << "memref<";
    for (int64_t d : shape) {
        oss << d << "x";
    }
    oss << dtype_mlir_name(dt) << ", #tpu.memory_space<vmem>>";
    return oss.str();
}

std::string format_vector_type(const std::vector<int64_t>& shape, DType dt) {
    std::ostringstream oss;
    oss << "vector<";
    for (int64_t d : shape) {
        oss << d << "x";
    }
    oss << dtype_mlir_name(dt) << ">";
    return oss.str();
}

uint64_t fnv1a_64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

class MosaicBuilder {
public:
    std::string next_val() { return "%" + std::to_string(next_id_++); }

    std::ostringstream& body() { return body_; }

    void emit_const_index(const std::string& name, int64_t v) {
        body_ << "    " << name << " = \"stable_mosaic.arith.constant\"() {value = " << v
              << " : index} : () -> index\n";
    }

    void emit_const_i32(const std::string& name, int64_t v) {
        body_ << "    " << name << " = \"stable_mosaic.arith.constant\"() {value = " << v
              << " : i32} : () -> i32\n";
    }

    void emit_const_zero_vec(const std::string& name, const std::vector<int64_t>& shape, DType dt) {
        const std::string vt = format_vector_type(shape, dt);
        body_ << "    " << name << " = \"stable_mosaic.arith.constant\"() {value = dense<0.000000e+00> : " << vt
              << "} : () -> " << vt << "\n";
    }

    void emit_vector_load_2d(const std::string& out,
                             const std::string& memref_arg,
                             const std::string& c0,
                             const std::vector<int64_t>& shape,
                             DType dt) {
        body_ << "    " << out << " = \"stable_mosaic.vector.load\"(" << memref_arg << ", " << c0 << ", " << c0
              << ") : (" << format_memref_vmem(shape, dt) << ", index, index) -> " << format_vector_type(shape, dt)
              << "\n";
    }

    void emit_vector_store_2d(const std::string& val,
                              const std::string& memref_arg,
                              const std::string& c0,
                              const std::vector<int64_t>& shape,
                              DType dt) {
        body_ << "    \"stable_mosaic.tpu.vector_store\"(" << val << ", " << memref_arg << ", " << c0 << ", " << c0
              << ") {add = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, strides = array<i32>} : ("
              << format_vector_type(shape, dt) << ", " << format_memref_vmem(shape, dt) << ", index, index) -> ()\n";
    }

    void emit_tpu_matmul(const std::string& out,
                         const std::string& lhs,
                         const std::string& rhs,
                         const std::string& acc_zero,
                         int64_t m,
                         int64_t k,
                         int64_t n) {
        const std::string lt = format_vector_type({m, k}, DType::BF16);
        const std::string rt = format_vector_type({k, n}, DType::BF16);
        const std::string ot = format_vector_type({m, n}, DType::F32);
        body_ << "    " << out << " = \"stable_mosaic.tpu.matmul\"(" << lhs << ", " << rhs << ", " << acc_zero
              << ") {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, "
              << "transpose_lhs = false, transpose_lhs_hint = false, transpose_rhs = false} : (" << lt << ", " << rt
              << ", " << ot << ") -> " << ot << "\n";
    }

private:
    std::ostringstream body_;
    int next_id_ = 0;
};

}  // namespace

std::string base64_encode(const std::string& input) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((input.size() + 2) / 3) * 4);
    const size_t n = input.size();
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t b0 = static_cast<unsigned char>(input[i]);
        const uint32_t b1 = (i + 1 < n) ? static_cast<unsigned char>(input[i + 1]) : 0U;
        const uint32_t b2 = (i + 2 < n) ? static_cast<unsigned char>(input[i + 2]) : 0U;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(triple >> 18) & 0x3F]);
        out.push_back(kAlphabet[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < n) ? kAlphabet[(triple >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < n) ? kAlphabet[triple & 0x3F] : '=');
    }
    return out;
}

std::string escape_mlir_string_literal(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 16);
    for (unsigned char c : raw) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

MosaicEmitResult emit_mosaic_kernel(const TileKernel& kernel, int64_t mosaic_version_override) {
    std::string err;
    if (!kernel.validate(&err)) {
        throw std::runtime_error("emit_mosaic_kernel: invalid TileKernel: " + err);
    }
    const TargetProfile prof = target_profile(kernel.target);
    if (!prof.supports_mosaic) {
        throw std::runtime_error(std::string("emit_mosaic_kernel: target ") + prof.name +
                                 " does not support Mosaic tpu_custom_call");
    }

    const int64_t version = (mosaic_version_override > 0) ? mosaic_version_override : prof.mosaic_version;
    const int64_t bm = kernel.grid.bm;
    const int64_t bn = kernel.grid.bn;
    const int64_t bk = kernel.grid.bk;
    const int64_t blk = kernel.matmul.quant_block_size;
    const bool is_quant = (kernel.matmul.quant != QuantFormat::NONE);
    const bool has_min = kernel.matmul.has_min;
    const bool is_multi_k = (kernel.grid.dims.size() == 2);
    const int64_t num_k = is_multi_k ? kernel.grid.dims[1] : 1;
    const int64_t nb_tile = is_quant ? (bk / blk) : 0;

    // Assign ^bb0 argument names.
    // 1D grid: %arg0 (j), then operand memrefs.
    // 2D grid: %arg0 (j), %arg1 (k), then operand memrefs, then scratch memref.
    int arg_counter = 0;
    const std::string arg_j = "%arg" + std::to_string(arg_counter++);
    const std::string arg_k = is_multi_k ? ("%arg" + std::to_string(arg_counter++)) : "";
    const std::string arg_x = "%arg" + std::to_string(arg_counter++);
    const std::string arg_w = "%arg" + std::to_string(arg_counter++);
    const std::string arg_d = is_quant ? ("%arg" + std::to_string(arg_counter++)) : "";
    const std::string arg_m = (is_quant && has_min) ? ("%arg" + std::to_string(arg_counter++)) : "";
    const std::string arg_o = "%arg" + std::to_string(arg_counter++);
    const std::string arg_acc = is_multi_k ? ("%arg" + std::to_string(arg_counter++)) : "";

    std::vector<std::string> bb0_params;
    std::vector<std::string> bb0_types;
    bb0_params.push_back(arg_j + ": i32");
    bb0_types.push_back("i32");
    if (is_multi_k) {
        bb0_params.push_back(arg_k + ": i32");
        bb0_types.push_back("i32");
    }
    for (const SramTile& op : kernel.operands) {
        const std::string mt = format_memref_vmem(op.window_bounds, op.dtype);
        bb0_types.push_back(mt);
    }
    for (const SramTile& sc : kernel.scratch) {
        bb0_types.push_back(format_memref_vmem(sc.window_bounds, sc.dtype));
    }
    {
        size_t idx = is_multi_k ? 2 : 1;
        bb0_params.push_back(arg_x + ": " + bb0_types[idx++]);
        bb0_params.push_back(arg_w + ": " + bb0_types[idx++]);
        if (is_quant) {
            bb0_params.push_back(arg_d + ": " + bb0_types[idx++]);
            if (has_min) {
                bb0_params.push_back(arg_m + ": " + bb0_types[idx++]);
            }
        }
        bb0_params.push_back(arg_o + ": " + bb0_types[idx++]);
        if (is_multi_k) {
            bb0_params.push_back(arg_acc + ": " + bb0_types[idx++]);
        }
    }

    MosaicBuilder b;
    const std::string c0 = b.next_val();
    b.emit_const_index(c0, 0);

    const std::string x_vec = b.next_val();
    b.emit_vector_load_2d(x_vec, arg_x, c0, {bm, bk}, DType::BF16);

    const std::string w_raw = b.next_val();
    b.emit_vector_load_2d(w_raw, arg_w, c0, {bk, bn}, kernel.matmul.weight_dtype);

    std::string w_bf16 = w_raw;
    if (kernel.matmul.weight_dtype != DType::BF16) {
        w_bf16 = b.next_val();
        const char* cvt_op = (kernel.matmul.weight_dtype == DType::U4 || kernel.matmul.weight_dtype == DType::U8)
                                 ? "stable_mosaic.arith.uitofp"
                                 : "stable_mosaic.arith.sitofp";
        b.body() << "    " << w_bf16 << " = \"" << cvt_op << "\"(" << w_raw << ") : ("
                 << format_vector_type({bk, bn}, kernel.matmul.weight_dtype) << ") -> "
                 << format_vector_type({bk, bn}, DType::BF16) << "\n";
        if (kernel.matmul.quant == QuantFormat::Q4_0 && kernel.matmul.unsigned_q4) {
            const std::string eight_s = b.next_val();
            const std::string eight_v = b.next_val();
            const std::string w_sub = b.next_val();
            const std::string w_vt = format_vector_type({bk, bn}, DType::BF16);
            b.body() << "    " << eight_s << " = \"stable_mosaic.arith.constant\"() {value = 8.000000e+00 : bf16} : () -> bf16\n";
            b.body() << "    " << eight_v << " = \"stable_mosaic.vector.broadcast\"(" << eight_s << ") : (bf16) -> " << w_vt << "\n";
            b.body() << "    " << w_sub << " = \"stable_mosaic.arith.subf\"(" << w_bf16 << ", " << eight_v
                     << ") {fastmath = #arith.fastmath<none>} : (" << w_vt << ", " << w_vt << ") -> " << w_vt << "\n";
            w_bf16 = w_sub;
        } else if (kernel.matmul.quant == QuantFormat::Q4_K && !kernel.matmul.unsigned_q4) {
            // Q4_K stores unshifted minimums m alongside signed S4 nibbles (q - 8 in [-8..7]);
            // adding 8.0 in BF16 restores exact unsigned nibbles [0..15] before scaling by d.
            const std::string eight_s = b.next_val();
            const std::string eight_v = b.next_val();
            const std::string w_add = b.next_val();
            const std::string w_vt = format_vector_type({bk, bn}, DType::BF16);
            b.body() << "    " << eight_s << " = \"stable_mosaic.arith.constant\"() {value = 8.000000e+00 : bf16} : () -> bf16\n";
            b.body() << "    " << eight_v << " = \"stable_mosaic.vector.broadcast\"(" << eight_s << ") : (bf16) -> " << w_vt << "\n";
            b.body() << "    " << w_add << " = \"stable_mosaic.arith.addf\"(" << w_bf16 << ", " << eight_v
                     << ") {fastmath = #arith.fastmath<none>} : (" << w_vt << ", " << w_vt << ") -> " << w_vt << "\n";
            w_bf16 = w_add;
        }
    }

    std::string tile_acc;
    if (!is_quant) {
        const std::string zero_acc = b.next_val();
        b.emit_const_zero_vec(zero_acc, {bm, bn}, DType::F32);
        tile_acc = b.next_val();
        b.emit_tpu_matmul(tile_acc, x_vec, w_bf16, zero_acc, bm, bk, bn);
    } else {
        const std::string d_vec = b.next_val();
        b.emit_vector_load_2d(d_vec, arg_d, c0, {nb_tile, bn}, DType::BF16);

        std::string m_vec;
        if (has_min) {
            m_vec = b.next_val();
            b.emit_vector_load_2d(m_vec, arg_m, c0, {nb_tile, bn}, DType::BF16);
        }

        const std::string out_f32_t = format_vector_type({bm, bn}, DType::F32);
        const std::string zero_acc = b.next_val();
        b.emit_const_zero_vec(zero_acc, {bm, bn}, DType::F32);
        std::string zero_row;
        if (has_min) {
            zero_row = b.next_val();
            b.emit_const_zero_vec(zero_row, {bm}, DType::F32);
        }
        tile_acc = zero_acc;

        for (int64_t blk_i = 0; blk_i < nb_tile; ++blk_i) {
            const int64_t k_off = blk_i * blk;
            const std::string x_blk = b.next_val();
            const std::string w_blk = b.next_val();
            const std::string d_blk_bf16 = b.next_val();
            const std::string d_blk_f32 = b.next_val();
            const std::string prod = b.next_val();
            const std::string d_bcast = b.next_val();
            const std::string scaled = b.next_val();

            b.body() << "    " << x_blk << " = \"stable_mosaic.vector.extract_strided_slice\"(" << x_vec
                     << ") {offsets = [0, " << k_off << "], sizes = [" << bm << ", " << blk << "], strides = [1, 1]} : ("
                     << format_vector_type({bm, bk}, DType::BF16) << ") -> "
                     << format_vector_type({bm, blk}, DType::BF16) << "\n";

            b.body() << "    " << w_blk << " = \"stable_mosaic.vector.extract_strided_slice\"(" << w_bf16
                     << ") {offsets = [" << k_off << ", 0], sizes = [" << blk << ", " << bn << "], strides = [1, 1]} : ("
                     << format_vector_type({bk, bn}, DType::BF16) << ") -> "
                     << format_vector_type({blk, bn}, DType::BF16) << "\n";

            b.body() << "    " << d_blk_bf16 << " = \"stable_mosaic.vector.extract_strided_slice\"(" << d_vec
                     << ") {offsets = [" << blk_i << ", 0], sizes = [1, " << bn << "], strides = [1, 1]} : ("
                     << format_vector_type({nb_tile, bn}, DType::BF16) << ") -> "
                     << format_vector_type({1, bn}, DType::BF16) << "\n";

            b.body() << "    " << d_blk_f32 << " = \"stable_mosaic.arith.extf\"(" << d_blk_bf16 << ") : ("
                     << format_vector_type({1, bn}, DType::BF16) << ") -> "
                     << format_vector_type({1, bn}, DType::F32) << "\n";

            b.emit_tpu_matmul(prod, x_blk, w_blk, zero_acc, bm, blk, bn);

            b.body() << "    " << d_bcast << " = \"stable_mosaic.vector.broadcast\"(" << d_blk_f32 << ") : ("
                     << format_vector_type({1, bn}, DType::F32) << ") -> " << out_f32_t << "\n";

            b.body() << "    " << scaled << " = \"stable_mosaic.arith.mulf\"(" << prod << ", " << d_bcast
                     << ") {fastmath = #arith.fastmath<none>} : (" << out_f32_t << ", " << out_f32_t << ") -> "
                     << out_f32_t << "\n";

            std::string blk_contrib = scaled;
            if (has_min) {
                // Keep activation block sum and minimum m in full F32 per sub-block (no BF16 truncf)
                // so exact cancellation (e.g. q*d + m == 0) is preserved before block reduction.
                const std::string x_blk_f32 = b.next_val();
                const std::string xsum_1d = b.next_val();
                const std::string xsum_2d = b.next_val();
                const std::string xsum_bcast = b.next_val();
                const std::string m_blk_bf16 = b.next_val();
                const std::string m_blk_f32 = b.next_val();
                const std::string m_bcast = b.next_val();
                const std::string m_term = b.next_val();
                const std::string scaled_plus_m = b.next_val();

                b.body() << "    " << x_blk_f32 << " = \"stable_mosaic.arith.extf\"(" << x_blk << ") : ("
                         << format_vector_type({bm, blk}, DType::BF16) << ") -> "
                         << format_vector_type({bm, blk}, DType::F32) << "\n";
                b.body() << "    " << xsum_1d << " = \"stable_mosaic.vector.multi_reduction\"(" << x_blk_f32 << ", "
                         << zero_row << ") {kind = #vector.kind<add>, reduction_dims = array<i64: 1>} : ("
                         << format_vector_type({bm, blk}, DType::F32) << ", "
                         << format_vector_type({bm}, DType::F32) << ") -> "
                         << format_vector_type({bm}, DType::F32) << "\n";
                b.body() << "    " << xsum_2d << " = \"stable_mosaic.vector.shape_cast\"(" << xsum_1d << ") : ("
                         << format_vector_type({bm}, DType::F32) << ") -> "
                         << format_vector_type({bm, 1}, DType::F32) << "\n";
                b.body() << "    " << xsum_bcast << " = \"stable_mosaic.vector.broadcast\"(" << xsum_2d << ") : ("
                         << format_vector_type({bm, 1}, DType::F32) << ") -> " << out_f32_t << "\n";
                b.body() << "    " << m_blk_bf16 << " = \"stable_mosaic.vector.extract_strided_slice\"(" << m_vec
                         << ") {offsets = [" << blk_i << ", 0], sizes = [1, " << bn << "], strides = [1, 1]} : ("
                         << format_vector_type({nb_tile, bn}, DType::BF16) << ") -> "
                         << format_vector_type({1, bn}, DType::BF16) << "\n";
                b.body() << "    " << m_blk_f32 << " = \"stable_mosaic.arith.extf\"(" << m_blk_bf16 << ") : ("
                         << format_vector_type({1, bn}, DType::BF16) << ") -> "
                         << format_vector_type({1, bn}, DType::F32) << "\n";
                b.body() << "    " << m_bcast << " = \"stable_mosaic.vector.broadcast\"(" << m_blk_f32 << ") : ("
                         << format_vector_type({1, bn}, DType::F32) << ") -> " << out_f32_t << "\n";
                b.body() << "    " << m_term << " = \"stable_mosaic.arith.mulf\"(" << xsum_bcast << ", " << m_bcast
                         << ") {fastmath = #arith.fastmath<none>} : (" << out_f32_t << ", " << out_f32_t << ") -> "
                         << out_f32_t << "\n";
                b.body() << "    " << scaled_plus_m << " = \"stable_mosaic.arith.addf\"(" << scaled << ", " << m_term
                         << ") {fastmath = #arith.fastmath<none>} : (" << out_f32_t << ", " << out_f32_t << ") -> "
                         << out_f32_t << "\n";
                blk_contrib = scaled_plus_m;
            }

            const std::string next_acc = b.next_val();
            b.body() << "    " << next_acc << " = \"stable_mosaic.arith.addf\"(" << tile_acc << ", " << blk_contrib
                     << ") {fastmath = #arith.fastmath<none>} : (" << out_f32_t << ", " << out_f32_t << ") -> "
                     << out_f32_t << "\n";
            tile_acc = next_acc;
        }
    }

    auto emit_store_final = [&](const std::string& final_f32, const std::string& indent) {
        std::string to_store = final_f32;
        if (kernel.matmul.out_dtype == DType::BF16) {
            to_store = b.next_val();
            b.body() << indent << to_store << " = \"stable_mosaic.arith.truncf\"(" << final_f32 << ") : ("
                     << format_vector_type({bm, bn}, DType::F32) << ") -> "
                     << format_vector_type({bm, bn}, DType::BF16) << "\n";
        }
        b.body() << indent << "\"stable_mosaic.tpu.vector_store\"(" << to_store << ", " << arg_o << ", " << c0 << ", "
                 << c0 << ") {add = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, strides = array<i32>} : ("
                 << format_vector_type({bm, bn}, kernel.matmul.out_dtype) << ", "
                 << format_memref_vmem({bm, bn}, kernel.matmul.out_dtype) << ", index, index) -> ()\n";
    };

    if (!is_multi_k) {
        emit_store_final(tile_acc, "    ");
    } else {
        const std::string c0_i32 = b.next_val();
        const std::string c_last_i32 = b.next_val();
        const std::string is_k0 = b.next_val();
        const std::string is_kgt0 = b.next_val();
        const std::string is_klast = b.next_val();

        b.emit_const_i32(c0_i32, 0);
        b.emit_const_i32(c_last_i32, num_k - 1);
        b.body() << "    " << is_k0 << " = \"stable_mosaic.arith.cmpi\"(" << arg_k << ", " << c0_i32
                 << ") {predicate = 0 : i64} : (i32, i32) -> i1\n";
        b.body() << "    \"stable_mosaic.scf.if\"(" << is_k0 << ") ({\n";
        b.body() << "      \"stable_mosaic.tpu.vector_store\"(" << tile_acc << ", " << arg_acc << ", " << c0 << ", "
                 << c0 << ") {add = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, strides = array<i32>} : ("
                 << format_vector_type({bm, bn}, DType::F32) << ", "
                 << format_memref_vmem({bm, bn}, DType::F32) << ", index, index) -> ()\n";
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }, {\n";
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }) : (i1) -> ()\n";

        b.body() << "    " << is_kgt0 << " = \"stable_mosaic.arith.cmpi\"(" << arg_k << ", " << c0_i32
                 << ") {predicate = 4 : i64} : (i32, i32) -> i1\n";
        const std::string prev_acc = b.next_val();
        const std::string sum_acc = b.next_val();
        b.body() << "    \"stable_mosaic.scf.if\"(" << is_kgt0 << ") ({\n";
        b.body() << "      " << prev_acc << " = \"stable_mosaic.vector.load\"(" << arg_acc << ", " << c0 << ", " << c0
                 << ") : (" << format_memref_vmem({bm, bn}, DType::F32) << ", index, index) -> "
                 << format_vector_type({bm, bn}, DType::F32) << "\n";
        b.body() << "      " << sum_acc << " = \"stable_mosaic.arith.addf\"(" << prev_acc << ", " << tile_acc
                 << ") {fastmath = #arith.fastmath<none>} : (" << format_vector_type({bm, bn}, DType::F32) << ", "
                 << format_vector_type({bm, bn}, DType::F32) << ") -> "
                 << format_vector_type({bm, bn}, DType::F32) << "\n";
        b.body() << "      \"stable_mosaic.tpu.vector_store\"(" << sum_acc << ", " << arg_acc << ", " << c0 << ", "
                 << c0 << ") {add = false, operandSegmentSizes = array<i32: 1, 1, 2, 0>, strides = array<i32>} : ("
                 << format_vector_type({bm, bn}, DType::F32) << ", "
                 << format_memref_vmem({bm, bn}, DType::F32) << ", index, index) -> ()\n";
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }, {\n";
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }) : (i1) -> ()\n";

        b.body() << "    " << is_klast << " = \"stable_mosaic.arith.cmpi\"(" << arg_k << ", " << c_last_i32
                 << ") {predicate = 0 : i64} : (i32, i32) -> i1\n";
        const std::string final_acc = b.next_val();
        b.body() << "    \"stable_mosaic.scf.if\"(" << is_klast << ") ({\n";
        b.body() << "      " << final_acc << " = \"stable_mosaic.vector.load\"(" << arg_acc << ", " << c0 << ", " << c0
                 << ") : (" << format_memref_vmem({bm, bn}, DType::F32) << ", index, index) -> "
                 << format_vector_type({bm, bn}, DType::F32) << "\n";
        emit_store_final(final_acc, "      ");
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }, {\n";
        b.body() << "      \"stable_mosaic.scf.yield\"() : () -> ()\n";
        b.body() << "    }) : (i1) -> ()\n";
    }

    b.body() << "    \"stable_mosaic.func.return\"() : () -> ()\n";

    std::ostringstream mod;
    mod << "module attributes {stable_mosaic.version = " << version << " : i64} {\n";
    mod << "  \"stable_mosaic.func.func\"() ({\n";
    mod << "  ^bb0(";
    for (size_t i = 0; i < bb0_params.size(); ++i) {
        if (i > 0) mod << ", ";
        mod << bb0_params[i];
    }
    mod << "):\n";
    mod << b.body().str();
    mod << "  }) {dimension_semantics = [";
    for (size_t i = 0; i < kernel.grid.semantics.size(); ++i) {
        if (i > 0) mod << ", ";
        mod << ((kernel.grid.semantics[i] == DimensionSemantics::PARALLEL)
                    ? "#tpu.dimension_semantics<parallel>"
                    : "#tpu.dimension_semantics<arbitrary>");
    }
    mod << "], function_type = (";
    for (size_t i = 0; i < bb0_types.size(); ++i) {
        if (i > 0) mod << ", ";
        mod << bb0_types[i];
    }
    mod << ") -> (), iteration_bounds = array<i64: ";
    for (size_t i = 0; i < kernel.grid.dims.size(); ++i) {
        if (i > 0) mod << ", ";
        mod << kernel.grid.dims[i];
    }
    mod << ">, scalar_prefetch = 0 : i64, scratch_operands = " << kernel.scratch.size()
        << " : i64, sym_name = \"" << kernel.name << "\", tpu.core_type = #tpu.core_type<tc>, window_params = [";
    for (size_t i = 0; i < kernel.operands.size(); ++i) {
        if (i > 0) mod << ", ";
        const auto& wb = kernel.operands[i].window_bounds;
        mod << "{transform_indices = @transform_" << i << ", window_bounds = array<i64: " << wb[0] << ", " << wb[1]
            << ">}";
    }
    mod << "]} : () -> ()\n";

    // Emit @transform_0 .. @transform_{N-1} by validated operand position (0 = activation, last = output)
    for (size_t i = 0; i < kernel.operands.size(); ++i) {
        const bool is_act = (i == 0);
        const bool is_out = (i + 1 == kernel.operands.size());
        mod << "  \"stable_mosaic.func.func\"() ({\n";
        if (!is_multi_k) {
            mod << "  ^bb0(%arg0: i32):\n";
            mod << "    %0 = \"stable_mosaic.arith.constant\"() {value = 0 : i32} : () -> i32\n";
            if (is_act) {
                mod << "    \"stable_mosaic.func.return\"(%0, %0) : (i32, i32) -> ()\n";
            } else {
                mod << "    \"stable_mosaic.func.return\"(%0, %arg0) : (i32, i32) -> ()\n";
            }
            mod << "  }) {function_type = (i32) -> (i32, i32), sym_name = \"transform_" << i << "\"} : () -> ()\n";
        } else {
            mod << "  ^bb0(%arg0: i32, %arg1: i32):\n";
            mod << "    %0 = \"stable_mosaic.arith.constant\"() {value = 0 : i32} : () -> i32\n";
            if (is_act) {
                mod << "    \"stable_mosaic.func.return\"(%0, %arg1) : (i32, i32) -> ()\n";
            } else if (is_out) {
                mod << "    \"stable_mosaic.func.return\"(%0, %arg0) : (i32, i32) -> ()\n";
            } else {
                mod << "    \"stable_mosaic.func.return\"(%arg1, %arg0) : (i32, i32) -> ()\n";
            }
            mod << "  }) {function_type = (i32, i32) -> (i32, i32), sym_name = \"transform_" << i << "\"} : () -> ()\n";
        }
    }

    mod << "}\n";

    MosaicEmitResult res;
    res.mosaic_mlir = mod.str();
    res.mosaic_mlir_b64 = base64_encode(res.mosaic_mlir);
    res.backend_config_json =
        std::string("{\"custom_call_config\": {\"body\": \"") + res.mosaic_mlir_b64 +
        "\", \"serialization_format\": 1, \"needs_layout_passes\": true}}";
    res.kernel_name = kernel.name;

    std::ostringstream key;
    key << std::hex << std::setfill('0') << std::setw(16) << fnv1a_64(res.mosaic_mlir)
        << "_" << std::dec << res.mosaic_mlir.size() << "_v" << version;
    res.cache_key = key.str();
    return res;
}

}  // namespace freetile
