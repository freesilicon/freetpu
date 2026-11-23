#include "freetile/tile_ir.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

namespace freetile {

namespace {

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool fail(std::string* err, const std::string& msg) {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

size_t sat_mul_size(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return std::numeric_limits<size_t>::max();
    }
    return a * b;
}

size_t sat_add_size(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return std::numeric_limits<size_t>::max();
    }
    return a + b;
}

size_t tile_elements(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t prod = 1;
    for (int64_t d : shape) {
        if (d <= 0) {
            return 0;
        }
        prod = sat_mul_size(prod, static_cast<size_t>(d));
    }
    return prod;
}

}  // namespace

TargetProfile target_profile(ChipTarget target) {
    TargetProfile p;
    p.target = target;
    switch (target) {
        case ChipTarget::V5E:
            p.name = "v5e";
            p.mxu_m = 128;
            p.mxu_n = 128;
            p.sublane_align = 8;
            p.lane_align = 128;
            p.vmem_bytes = 16ULL * 1024ULL * 1024ULL;
            p.smem_bytes = 1024ULL * 1024ULL;
            p.supports_mosaic = true;
            p.supports_s4_native = true;
            p.supports_bf16 = true;
            p.mosaic_version = 17;
            break;
        case ChipTarget::V6E:
            p.name = "v6e";
            p.mxu_m = 256;
            p.mxu_n = 256;
            p.sublane_align = 8;
            p.lane_align = 128;
            p.vmem_bytes = 32ULL * 1024ULL * 1024ULL;
            p.smem_bytes = 1024ULL * 1024ULL;
            p.supports_mosaic = true;
            p.supports_s4_native = true;
            p.supports_bf16 = true;
            p.mosaic_version = 17;
            break;
        case ChipTarget::TPU7X:
            p.name = "tpu7x";
            p.mxu_m = 256;
            p.mxu_n = 256;
            p.sublane_align = 8;
            p.lane_align = 128;
            p.vmem_bytes = 64ULL * 1024ULL * 1024ULL;
            p.smem_bytes = 1024ULL * 1024ULL;
            p.supports_mosaic = true;
            p.supports_s4_native = true;
            p.supports_bf16 = true;
            p.mosaic_version = 17;
            break;
        case ChipTarget::BUENOS_A0:
            p.name = "buenos_a0";
            p.mxu_m = 64;
            p.mxu_n = 64;
            p.sublane_align = 8;
            p.lane_align = 64;
            p.vmem_bytes = 16ULL * 1024ULL * 1024ULL;
            p.smem_bytes = 512ULL * 1024ULL;
            p.supports_mosaic = false;
            p.supports_s4_native = false;
            p.supports_bf16 = true;
            p.mosaic_version = 0;
            break;
    }
    return p;
}

ChipTarget detect_chip_target(const std::string& device_kind) {
    const std::string lower = to_lower_ascii(device_kind);
    if (lower.find("7x") != std::string::npos || lower.find("ironwood") != std::string::npos) {
        return ChipTarget::TPU7X;
    }
    if (lower.find("v6") != std::string::npos || lower.find("trillium") != std::string::npos) {
        return ChipTarget::V6E;
    }
    if (lower.find("buenos") != std::string::npos || lower.find("tensor g5") != std::string::npos) {
        return ChipTarget::BUENOS_A0;
    }
    return ChipTarget::V5E;
}

const char* chip_target_name(ChipTarget target) {
    return target_profile(target).name;
}

const char* dtype_mlir_name(DType dt) {
    // Mosaic MLIR (memref/vector/arith) requires signless integer types; signedness
    // is carried by conversion ops (arith.sitofp vs arith.uitofp).
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::S4:
        case DType::U4: return "i4";
        case DType::S8:
        case DType::U8: return "i8";
        case DType::S32: return "i32";
        case DType::I1: return "i1";
        default: return "invalid";
    }
}

size_t dtype_bits(DType dt) {
    switch (dt) {
        case DType::I1: return 1;
        case DType::S4:
        case DType::U4: return 4;
        case DType::S8:
        case DType::U8: return 8;
        case DType::F16:
        case DType::BF16: return 16;
        case DType::F32:
        case DType::S32: return 32;
        default: return 0;
    }
}

size_t dtype_nbytes(DType dt, size_t num_elements) {
    const size_t bits = dtype_bits(dt);
    if (bits == 0 || num_elements == 0) {
        return 0;
    }
    const size_t total_bits = sat_mul_size(num_elements, bits);
    if (total_bits == std::numeric_limits<size_t>::max()) {
        return std::numeric_limits<size_t>::max();
    }
    return sat_add_size(total_bits, 7) / 8;
}

const char* quant_format_name(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::NONE: return "dense";
        case QuantFormat::Q4_K: return "q4_k";
        case QuantFormat::Q6_K: return "q6_k";
        case QuantFormat::Q4_0: return "q4_0";
        case QuantFormat::Q8_0: return "q8_0";
        default: return "unknown";
    }
}

int64_t quant_block_size(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::NONE: return 1;
        case QuantFormat::Q4_K:
        case QuantFormat::Q4_0:
        case QuantFormat::Q8_0: return 32;
        case QuantFormat::Q6_K: return 16;
        default: return 0;
    }
}

bool quant_has_min(QuantFormat fmt) {
    return fmt == QuantFormat::Q4_K;
}

const char* memory_space_name(MemorySpace space) {
    switch (space) {
        case MemorySpace::HBM: return "hbm";
        case MemorySpace::VMEM: return "vmem";
        case MemorySpace::SMEM: return "smem";
        case MemorySpace::SRAM: return "sram";
        default: return "unknown";
    }
}

size_t TileKernel::vmem_footprint_bytes() const {
    size_t total = 0;
    for (const SramTile& op : operands) {
        if (op.space == MemorySpace::VMEM || op.space == MemorySpace::SRAM) {
            size_t tile_bytes = dtype_nbytes(op.dtype, tile_elements(op.window_bounds));
            bool double_buf = false;
            for (const DmaPrefetch& pf : prefetches) {
                if (pf.operand_name == op.name && pf.double_buffered) {
                    double_buf = true;
                    break;
                }
            }
            total = sat_add_size(total, double_buf ? sat_mul_size(tile_bytes, 2) : tile_bytes);
        }
    }
    for (const SramTile& sc : scratch) {
        if (sc.space == MemorySpace::VMEM || sc.space == MemorySpace::SRAM) {
            total = sat_add_size(total, dtype_nbytes(sc.dtype, tile_elements(sc.window_bounds)));
        }
    }
    // Account for unpacked BF16 weight tile when weight is S4/U4/S8.
    if (matmul.quant != QuantFormat::NONE && grid.bk > 0 && grid.bn > 0 &&
        (matmul.weight_dtype == DType::S4 || matmul.weight_dtype == DType::U4 ||
         matmul.weight_dtype == DType::S8)) {
        const size_t elems = sat_mul_size(static_cast<size_t>(grid.bk), static_cast<size_t>(grid.bn));
        total = sat_add_size(total, dtype_nbytes(DType::BF16, elems));
    }
    return total;
}

bool TileKernel::validate(std::string* err) const {
    if (name.empty()) {
        return fail(err, "TileKernel name must not be empty");
    }
    const TargetProfile prof = target_profile(target);
    if (matmul.m <= 0 || matmul.m_padded <= 0 || matmul.n <= 0 || matmul.k <= 0) {
        return fail(err, "TileMatmul dimensions must be positive");
    }
    if (matmul.m_padded < matmul.m || (matmul.m_padded % prof.sublane_align) != 0) {
        return fail(err, "TileMatmul m_padded must be >= m and aligned to sublane_align");
    }
    if (grid.bm != matmul.m_padded) {
        return fail(err, "TileGrid bm must equal matmul.m_padded");
    }
    if (grid.bn <= 0 || (grid.bn % prof.lane_align) != 0 || (matmul.n % grid.bn) != 0) {
        return fail(err, "TileGrid bn must be positive, lane-aligned, and evenly divide N");
    }
    if (grid.bk <= 0 || (matmul.k % grid.bk) != 0) {
        return fail(err, "TileGrid bk must be positive and evenly divide K");
    }

    const int64_t blk = quant_block_size(matmul.quant);
    if (blk <= 0 || matmul.quant_block_size != blk) {
        return fail(err, "TileMatmul quant_block_size does not match QuantFormat");
    }
    if ((grid.bk % blk) != 0) {
        return fail(err, "TileGrid bk must be divisible by quant_block_size");
    }
    if (matmul.quant != QuantFormat::NONE) {
        const int64_t nb_tile = grid.bk / blk;
        if ((nb_tile % prof.sublane_align) != 0) {
            return fail(err, "Quantized tile block count (bk / blk) must be divisible by sublane_align (8)");
        }
    } else if ((grid.bk % prof.sublane_align) != 0) {
        return fail(err, "Dense tile bk must be divisible by sublane_align (8)");
    }

    const size_t n_tiles = static_cast<size_t>(matmul.n / grid.bn);
    const size_t k_tiles = static_cast<size_t>(matmul.k / grid.bk);
    if (k_tiles == 1) {
        if (grid.dims.size() != 1 || grid.dims[0] != static_cast<int64_t>(n_tiles)) {
            return fail(err, "Single-K TileGrid must be 1D {N/bn}");
        }
        if (!scratch.empty()) {
            return fail(err, "Single-K TileKernel must not allocate accumulator scratch");
        }
    } else {
        if (grid.dims.size() != 2 || grid.dims[0] != static_cast<int64_t>(n_tiles) ||
            grid.dims[1] != static_cast<int64_t>(k_tiles)) {
            return fail(err, "Multi-K TileGrid must be 2D {N/bn, K/bk}");
        }
        if (scratch.size() != 1 || !scratch[0].is_scratch) {
            return fail(err, "Multi-K TileKernel requires 1 F32 VMEM scratch accumulator");
        }
    }

    if (grid.semantics.size() != grid.dims.size()) {
        return fail(err, "TileGrid semantics size must match grid dims");
    }
    if (k_tiles > 1 && grid.semantics[1] != DimensionSemantics::ARBITRARY) {
        return fail(err, "Multi-K TileGrid reduction axis (dim 1) must use ARBITRARY semantics");
    }

    if (matmul.has_min != quant_has_min(matmul.quant)) {
        return fail(err, "TileMatmul has_min does not match QuantFormat");
    }
    const size_t expected_ops = (matmul.quant == QuantFormat::NONE) ? 3
                                : (matmul.has_min ? 5 : 4);
    if (operands.size() != expected_ops) {
        return fail(err, "TileKernel operand count mismatch for QuantFormat");
    }
    if (matmul.act_dtype != DType::BF16) {
        return fail(err, "TileMatmul act_dtype must be BF16");
    }
    if (matmul.out_dtype != DType::BF16 && matmul.out_dtype != DType::F32) {
        return fail(err, "TileMatmul out_dtype must be BF16 or F32");
    }
    if (matmul.quant == QuantFormat::NONE) {
        if (matmul.weight_dtype != DType::BF16) {
            return fail(err, "Dense TileMatmul weight_dtype must be BF16");
        }
    } else {
        if (matmul.scale_dtype != DType::BF16) {
            return fail(err, "Quantized TileMatmul scale_dtype must be BF16");
        }
        const DType expected_w_dt =
            (matmul.quant == QuantFormat::Q4_K || matmul.quant == QuantFormat::Q4_0)
                ? (matmul.unsigned_q4 ? DType::U4 : DType::S4)
                : DType::S8;
        if (matmul.weight_dtype != expected_w_dt) {
            return fail(err, "Quantized TileMatmul weight_dtype does not match QuantFormat/unsigned_q4");
        }
    }

    auto check_operand = [&](size_t idx,
                             const char* label,
                             int64_t g0,
                             int64_t g1,
                             int64_t w0,
                             int64_t w1,
                             DType expected_dt,
                             bool expected_output) -> bool {
        const SramTile& op = operands[idx];
        if (op.window_bounds.size() != 2 || op.global_shape.size() != 2) {
            return fail(err, std::string("SramTile operand ") + label + " must be 2D");
        }
        if (op.global_shape[0] != g0 || op.global_shape[1] != g1) {
            return fail(err, std::string("SramTile operand ") + label + " global_shape does not match matmul dims");
        }
        if (op.window_bounds[0] != w0 || op.window_bounds[1] != w1) {
            return fail(err, std::string("SramTile operand ") + label + " window_bounds does not match grid tile dims");
        }
        if (op.dtype != expected_dt) {
            return fail(err, std::string("SramTile operand ") + label + " dtype does not match TileMatmul");
        }
        if (op.is_output != expected_output) {
            return fail(err, std::string("SramTile operand ") + label + " is_output flag mismatch");
        }
        return true;
    };

    size_t op_idx = 0;
    if (!check_operand(op_idx++, "x", matmul.m_padded, matmul.k, grid.bm, grid.bk, matmul.act_dtype, false)) {
        return false;
    }
    if (!check_operand(op_idx++, "w", matmul.k, matmul.n, grid.bk, grid.bn, matmul.weight_dtype, false)) {
        return false;
    }
    if (matmul.quant != QuantFormat::NONE) {
        const int64_t nb = matmul.k / blk;
        const int64_t nb_tile = grid.bk / blk;
        if (!check_operand(op_idx++, "d", nb, matmul.n, nb_tile, grid.bn, matmul.scale_dtype, false)) {
            return false;
        }
        if (matmul.has_min) {
            if (!check_operand(op_idx++, "m", nb, matmul.n, nb_tile, grid.bn, matmul.scale_dtype, false)) {
                return false;
            }
        }
    }
    if (!check_operand(op_idx++, "out", matmul.m_padded, matmul.n, grid.bm, grid.bn, matmul.out_dtype, true)) {
        return false;
    }

    for (const SramTile& sc : scratch) {
        if (sc.window_bounds.size() != 2 ||
            sc.window_bounds[0] != grid.bm || sc.window_bounds[1] != grid.bn) {
            return fail(err, "SramTile scratch window_bounds must match {bm, bn}");
        }
        if (sc.dtype != DType::F32) {
            return fail(err, "SramTile scratch accumulator dtype must be F32");
        }
    }
    if (vmem_footprint_bytes() > prof.vmem_bytes) {
        std::ostringstream oss;
        oss << "VMEM footprint (" << vmem_footprint_bytes() << " B) exceeds target budget ("
            << prof.vmem_bytes << " B)";
        return fail(err, oss.str());
    }
    return true;
}

bool plan_quant_matmul_kernel(ChipTarget target,
                              int64_t m,
                              int64_t k,
                              int64_t n,
                              QuantFormat fmt,
                              DType scale_dt,
                              DType out_dt,
                              bool unsigned_q4,
                              TileKernel* out_kernel,
                              std::string* reason,
                              const TilePlannerConfig* planner_cfg) {
    if (out_kernel == nullptr) {
        return fail(reason, "out_kernel is null");
    }
    if (m <= 0 || k <= 0 || n <= 0) {
        return fail(reason, "M, K, N must be positive");
    }
    const TargetProfile prof = target_profile(target);
    if (m > 64) {
        return fail(reason, "Decode/small-batch FreeTile kernel supports M <= 64");
    }
    if ((n % prof.lane_align) != 0) {
        return fail(reason, "N must be divisible by target lane_align");
    }
    if (out_dt != DType::BF16 && out_dt != DType::F32) {
        return fail(reason, "Output dtype must be BF16 or F32");
    }
    if (fmt != QuantFormat::NONE && scale_dt != DType::BF16 && scale_dt != DType::F32 && scale_dt != DType::F16) {
        return fail(reason, "Scale dtype must be BF16, F16, or F32");
    }

    const int64_t m_pad = ((m + prof.sublane_align - 1) / prof.sublane_align) * prof.sublane_align;
    const int64_t blk = quant_block_size(fmt);
    if (blk <= 0) {
        return fail(reason, "Unsupported QuantFormat");
    }
    if ((k % blk) != 0) {
        return fail(reason, "K must be divisible by quant block size");
    }
    const int64_t min_bk = (fmt == QuantFormat::NONE) ? prof.sublane_align : (blk * prof.sublane_align);
    if ((k % min_bk) != 0) {
        return fail(reason, (fmt == QuantFormat::NONE)
                                ? "Dense K must be divisible by sublane_align (8)"
                                : "K must be divisible by quant_block_size * 8");
    }

    std::vector<int64_t> bn_cands = {2048, 1024, 512, 256, 128, 64};
    std::vector<int64_t> bk_cands = {2048, 1024, 512, 256, 128};
    if (fmt == QuantFormat::NONE) {
        if (k < 128 && k >= min_bk && (k % min_bk) == 0) {
            bk_cands.push_back(k);
        }
        bk_cands.push_back(64);
        bk_cands.push_back(32);
        bk_cands.push_back(16);
        bk_cands.push_back(8);
    }
    int64_t override_bm = m_pad;
    if (planner_cfg != nullptr && planner_cfg->schedule_hook != nullptr) {
        const TileShapeOverride ov = planner_cfg->schedule_hook(
            target, m, k, n, fmt, scale_dt, out_dt, unsigned_q4, planner_cfg->hook_user_data);
        if (ov.active) {
            if (ov.bm >= m && ov.bm <= 1024 && (ov.bm % prof.sublane_align) == 0) {
                override_bm = ov.bm;
            }
            if (ov.bn > 0) {
                bn_cands.insert(bn_cands.begin(), ov.bn);
            }
            if (ov.bk > 0) {
                bk_cands.insert(bk_cands.begin(), ov.bk);
            }
        }
    }

    DType w_dt = DType::BF16;
    if (fmt == QuantFormat::Q4_K || fmt == QuantFormat::Q4_0) {
        w_dt = unsigned_q4 ? DType::U4 : DType::S4;
    } else if (fmt == QuantFormat::Q6_K || fmt == QuantFormat::Q8_0) {
        w_dt = DType::S8;
    }

    // Reserve VMEM headroom for Mosaic compiler temporaries and DMA descriptors.
    // Default 85% admits 2048x2048 S4 (~13.2 MiB / 16 MiB) while keeping S8 at 1024x2048 (~8.5 MiB).
    const int64_t vmem_pct =
        (planner_cfg != nullptr && planner_cfg->max_vmem_pct > 0 && planner_cfg->max_vmem_pct <= 100)
            ? planner_cfg->max_vmem_pct
            : 85;
    const size_t safe_vmem_budget = (prof.vmem_bytes * static_cast<size_t>(vmem_pct)) / 100;

    bool found = false;
    TileKernel best;
    for (int64_t bn : bn_cands) {
        if (bn < prof.lane_align || (bn % prof.lane_align) != 0 || (n % bn) != 0) {
            continue;
        }
        for (int64_t bk : bk_cands) {
            if (bk < min_bk || (bk % min_bk) != 0 || (k % bk) != 0 || (bk % blk) != 0) {
                continue;
            }
            if (fmt != QuantFormat::NONE && ((bk / blk) % prof.sublane_align) != 0) {
                continue;
            }

            TileKernel cand;
            std::ostringstream kname;
            kname << "ft_" << quant_format_name(fmt) << "_m" << override_bm << "_k" << k << "_n" << n
                  << "_bk" << bk << "_bn" << bn;
            cand.name = kname.str();
            cand.target = target;
            cand.preferred_lowering = (target == ChipTarget::BUENOS_A0)
                                          ? LoweringTarget::OPENBUENOS_DGC0_HOOK
                                          : LoweringTarget::MOSAIC_TPU_CUSTOM_CALL;

            const int64_t grid_n = n / bn;
            const int64_t grid_k = k / bk;
            cand.grid.bm = override_bm;
            cand.grid.bn = bn;
            cand.grid.bk = bk;
            if (grid_k == 1) {
                cand.grid.dims = {grid_n};
                cand.grid.semantics = {DimensionSemantics::PARALLEL};
            } else {
                cand.grid.dims = {grid_n, grid_k};
                cand.grid.semantics = {DimensionSemantics::PARALLEL, DimensionSemantics::ARBITRARY};
            }

            cand.matmul.m = m;
            cand.matmul.m_padded = override_bm;
            cand.matmul.n = n;
            cand.matmul.k = k;
            cand.matmul.quant = fmt;
            cand.matmul.quant_block_size = blk;
            cand.matmul.has_min = quant_has_min(fmt);
            cand.matmul.unsigned_q4 = unsigned_q4;
            cand.matmul.act_dtype = DType::BF16;
            cand.matmul.weight_dtype = w_dt;
            cand.matmul.scale_dtype = DType::BF16;
            cand.matmul.out_dtype = out_dt;
            cand.matmul.factored_m_term = (fmt == QuantFormat::Q4_K);
            cand.matmul.post_dot_scale = (fmt != QuantFormat::NONE);

            const MemorySpace sram_space =
                (target == ChipTarget::BUENOS_A0) ? MemorySpace::SRAM : MemorySpace::VMEM;

            cand.operands.push_back({"x", sram_space, DType::BF16, {override_bm, k}, {override_bm, bk}, false, false});
            cand.operands.push_back({"w", sram_space, w_dt, {k, n}, {bk, bn}, false, false});
            if (fmt != QuantFormat::NONE) {
                const int64_t nb = k / blk;
                const int64_t nb_tile = bk / blk;
                cand.operands.push_back({"d", sram_space, DType::BF16, {nb, n}, {nb_tile, bn}, false, false});
                if (cand.matmul.has_min) {
                    cand.operands.push_back({"m", sram_space, DType::BF16, {nb, n}, {nb_tile, bn}, false, false});
                }
            }
            cand.operands.push_back({"out", sram_space, out_dt, {override_bm, n}, {override_bm, bn}, true, false});

            if (grid_k > 1) {
                cand.scratch.push_back({"acc", sram_space, DType::F32, {override_bm, n}, {override_bm, bn}, false, true});
            }

            const bool db = (grid_k > 1);
            cand.prefetches.push_back({"x", MemorySpace::HBM, sram_space, {override_bm, bk}, db, db ? 2 : 1,
                                       db ? SyncMode::ASYNC_PIPELINE : SyncMode::SYNCHRONOUS});
            cand.prefetches.push_back({"w", MemorySpace::HBM, sram_space, {bk, bn}, db, db ? 2 : 1,
                                       db ? SyncMode::ASYNC_PIPELINE : SyncMode::SYNCHRONOUS});
            if (fmt != QuantFormat::NONE) {
                const int64_t nb_tile = bk / blk;
                cand.prefetches.push_back({"d", MemorySpace::HBM, sram_space, {nb_tile, bn}, db, db ? 2 : 1,
                                           db ? SyncMode::ASYNC_PIPELINE : SyncMode::SYNCHRONOUS});
                if (cand.matmul.has_min) {
                    cand.prefetches.push_back({"m", MemorySpace::HBM, sram_space, {nb_tile, bn}, db, db ? 2 : 1,
                                               db ? SyncMode::ASYNC_PIPELINE : SyncMode::SYNCHRONOUS});
                }
            }

            std::string val_err;
            if (cand.validate(&val_err) && cand.vmem_footprint_bytes() <= safe_vmem_budget) {
                best = std::move(cand);
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }

    if (!found) {
        return fail(reason, "No valid FreeTile (bk, bn) tiling fits target alignment and VMEM budget");
    }

    *out_kernel = std::move(best);
    return true;
}

}  // namespace freetile
