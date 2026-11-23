#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace freetile {

enum class ChipTarget {
    V5E = 0,
    V6E,
    TPU7X,
    BUENOS_A0,
};

enum class LoweringTarget {
    MOSAIC_TPU_CUSTOM_CALL = 0,
    OPENTPU_VFIO_HOOK,
    OPENBUENOS_DGC0_HOOK,
};

struct TargetProfile {
    ChipTarget target = ChipTarget::V5E;
    const char* name = "v5e";
    int64_t mxu_m = 128;
    int64_t mxu_n = 128;
    int64_t sublane_align = 8;
    int64_t lane_align = 128;
    size_t vmem_bytes = 16ULL * 1024ULL * 1024ULL;
    size_t smem_bytes = 1024ULL * 1024ULL;
    bool supports_mosaic = true;
    bool supports_s4_native = true;
    bool supports_bf16 = true;
    int64_t mosaic_version = 17;
};

TargetProfile target_profile(ChipTarget target);
ChipTarget detect_chip_target(const std::string& device_kind);
const char* chip_target_name(ChipTarget target);

enum class DType {
    INVALID = 0,
    F32,
    F16,
    BF16,
    S4,
    U4,
    S8,
    U8,
    S32,
    I1,
};

const char* dtype_mlir_name(DType dt);
size_t dtype_bits(DType dt);
size_t dtype_nbytes(DType dt, size_t num_elements);

enum class QuantFormat {
    NONE = 0,
    Q4_K,
    Q6_K,
    Q4_0,
    Q8_0,
};

const char* quant_format_name(QuantFormat fmt);
int64_t quant_block_size(QuantFormat fmt);
bool quant_has_min(QuantFormat fmt);

enum class MemorySpace {
    HBM = 0,
    VMEM,
    SMEM,
    SRAM,  // Portable alias for edge / direct-VFIO SRAM targets (BUENOS_A0 / OPENTPU)
};

const char* memory_space_name(MemorySpace space);

enum class DimensionSemantics {
    PARALLEL = 0,
    ARBITRARY,
};

enum class SyncMode {
    SYNCHRONOUS = 0,
    ASYNC_PIPELINE,
};

struct TileGrid {
    std::vector<int64_t> dims;                    // e.g. {N/bn, K/bk} or {N/bn}
    std::vector<DimensionSemantics> semantics;    // parallel / arbitrary per grid axis
    int64_t bm = 8;
    int64_t bn = 128;
    int64_t bk = 256;
};

struct SramTile {
    std::string name;
    MemorySpace space = MemorySpace::VMEM;
    DType dtype = DType::BF16;
    std::vector<int64_t> global_shape;
    std::vector<int64_t> window_bounds;
    bool is_output = false;
    bool is_scratch = false;
};

struct DmaPrefetch {
    std::string operand_name;
    MemorySpace src_space = MemorySpace::HBM;
    MemorySpace dst_space = MemorySpace::VMEM;
    std::vector<int64_t> tile_shape;
    bool double_buffered = true;
    int32_t stages = 2;
    SyncMode sync_mode = SyncMode::SYNCHRONOUS;
};

struct TileMatmul {
    int64_t m = 1;
    int64_t m_padded = 8;
    int64_t n = 128;
    int64_t k = 256;
    QuantFormat quant = QuantFormat::Q4_K;
    int64_t quant_block_size = 32;
    bool has_min = true;
    bool unsigned_q4 = false;
    DType act_dtype = DType::BF16;
    DType weight_dtype = DType::S4;
    DType scale_dtype = DType::BF16;
    DType out_dtype = DType::BF16;
    bool factored_m_term = true;
    bool post_dot_scale = true;
};

struct TileKernel {
    std::string name;
    ChipTarget target = ChipTarget::V5E;
    LoweringTarget preferred_lowering = LoweringTarget::MOSAIC_TPU_CUSTOM_CALL;
    TileGrid grid;
    std::vector<SramTile> operands;
    std::vector<SramTile> scratch;
    std::vector<DmaPrefetch> prefetches;
    TileMatmul matmul;

    size_t vmem_footprint_bytes() const;
    bool validate(std::string* err = nullptr) const;
};

// Optional extension hook/config seam so external runtime or tuning layers
// can override tile selection without forking FreeTile core IR.
struct TileShapeOverride {
    bool active = false;
    int64_t bm = 0;
    int64_t bn = 0;
    int64_t bk = 0;
};

using TileScheduleHookFn = TileShapeOverride (*)(ChipTarget target,
                                                 int64_t m,
                                                 int64_t k,
                                                 int64_t n,
                                                 QuantFormat fmt,
                                                 DType scale_dt,
                                                 DType out_dt,
                                                 bool unsigned_q4,
                                                 void* user_data);

struct TilePlannerConfig {
    TileScheduleHookFn schedule_hook = nullptr;
    void* hook_user_data = nullptr;
    int64_t max_vmem_pct = 85;  // Max fraction of target VMEM budget to consume
};

// Plans an aligned, VMEM-feasible quantized or dense TileKernel for the given target.
// Returns false with a diagnostic reason if the shape/dtype/target is ineligible.
bool plan_quant_matmul_kernel(ChipTarget target,
                              int64_t m,
                              int64_t k,
                              int64_t n,
                              QuantFormat fmt,
                              DType scale_dt,
                              DType out_dt,
                              bool unsigned_q4,
                              TileKernel* out_kernel,
                              std::string* reason = nullptr,
                              const TilePlannerConfig* planner_cfg = nullptr);

}  // namespace freetile
