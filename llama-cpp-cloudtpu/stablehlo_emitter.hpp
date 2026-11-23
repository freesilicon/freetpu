#pragma once

#include "pjrt_runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cloudtpu {

enum class TensorStorageKind {
    DENSE = 0,
    Q8_0_SPLIT,  // K-major N-minor: (qs: [B3, B2, K/32, 32, N] S8, d: [B3, B2, K/32, N] F16)
    Q4_0_SPLIT,  // K-major N-minor: (qs: [B3, B2, K/32, 16, N] U8, d: [B3, B2, K/32, N] F16)
    Q4_K_SPLIT,  // K-major N-minor: (qs: [B3, B2, K/32, 16, N] U8, d_eff/m_eff: [B3, B2, K/32, N] F32/BF16)
    Q6_K_SPLIT,  // K-major N-minor: (qs: [B3, B2, K/16, 16, N] S8, d_eff: [B3, B2, K/16, N] F32/BF16)
};

enum class TensorShardingKind {
    REPLICATED = 0,
    SHARD_N,  // Sharded evenly across spmd_chips along StableHLO dim 2 (GGML ne[1] rows)
    SHARD_K,  // Sharded evenly across spmd_chips along StableHLO dim 3 (GGML ne[0] cols)
};

// Row-major tensor descriptor (GGML ne[0..3] maps to row-major dims[0..3] = {ne3, ne2, ne1, ne0}).
struct TensorSpec {
    uint64_t id = 0;  // Unique tensor slot id inside a GraphProgramSpec
    PjrtDType dtype = PjrtDType::F32;
    TensorStorageKind storage = TensorStorageKind::DENSE;
    int64_t dims[4] = {1, 1, 1, 1};  // Logical (global) shape [ne3, ne2, ne1, ne0]
    TensorShardingKind sharding = TensorShardingKind::REPLICATED;
    int32_t alias_output_idx = -1;  // >= 0 emits {tf.aliasing_output = idx : i32} for in-place HBM donation
    PjrtDType scale_dtype = PjrtDType::INVALID;  // Optional override for Q4_K/Q6_K scale/min dtype (INVALID = default)

    int64_t ne0() const { return dims[3]; }
    int64_t ne1() const { return dims[2]; }
    int64_t ne2() const { return dims[1]; }
    int64_t ne3() const { return dims[0]; }
};

enum class CloudOpKind {
    MUL_MAT = 0,
    RMS_NORM,
    ROPE,
    SILU,
    SWIGLU,
    SOFT_MAX,
    GET_ROWS,
    SET_ROWS,
    ADD,
    MUL,
    SCALE,
    CPY,
    RESHAPE,
    TRANSPOSE,
    SLICE,
    FLASH_ATTN_EXT,
    ALL_REDUCE_SUM,
    ALL_GATHER,
    REDUCE_SCATTER_SUM,
};

struct RopeParams {
    int32_t n_dims = 0;
    int32_t mode = 0;  // 0 = NORMAL, 2 = NEOX
    int32_t n_ctx_orig = 0;
    int32_t n_offs = 0;
    bool truncate = false;
    float freq_base = 10000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
};

struct SoftMaxParams {
    float scale = 1.0f;
    float max_bias = 0.0f;
};

struct FlashAttnParams {
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
};

struct OpNodeSpec {
    CloudOpKind kind = CloudOpKind::MUL_MAT;
    std::vector<uint64_t> src_ids;  // TensorSpec::id references (graph inputs or earlier op outputs)
    TensorSpec dst;

    // Op-specific parameters
    bool high_precision = false;  // MUL_MAT: GGML_PREC_F32
    bool keep_sharded_out = false;  // SPMD MUL_MAT(SHARD_N): keep output as SHARD_K (Megatron FFN)
    int64_t collective_dim = 3;     // ALL_GATHER / REDUCE_SCATTER axis
    float eps = 1e-5f;            // RMS_NORM
    float scale = 1.0f;           // SCALE
    float bias = 0.0f;            // SCALE
    int32_t glu_swapped = 0;      // SWIGLU single-input split swap
    int64_t perm[4] = {0, 1, 2, 3};  // TRANSPOSE (row-major StableHLO axis permutation)
    std::vector<int64_t> slice_reshape;  // SLICE: optional pre-slice reshape dims
    std::vector<int64_t> slice_starts;   // SLICE: start indices
    std::vector<int64_t> slice_limits;   // SLICE: exclusive end indices
    RopeParams rope;
    SoftMaxParams soft_max;
    FlashAttnParams flash_attn;
};

struct GraphProgramSpec {
    std::vector<TensorSpec> inputs;   // External live-in tensors
    std::vector<OpNodeSpec> ops;      // Topologically ordered ops
    std::vector<uint64_t> output_ids; // Live-out TensorSpec::ids
    int32_t spmd_chips = 1;           // Number of replica-SPMD chips (1 = single chip, 4 = tpu7x, 8 = v6e-8)
    bool bf16_matmul = true;          // Cast split-quantized weights & activations to BF16 for MXU (dense F16/F32 stay F32)
    bool bf16_flash_attn = false;     // Opt-in BF16 Q/K/probs/V matmuls in FLASH_ATTN_EXT (default false: F32 parity)
    bool q4_split_dot = true;         // Q4_0/Q4_K: true = two split-X dots (zero concat); false = single concat dot
};

struct PjrtInputBinding {
    uint64_t tensor_id = 0;
    int part = 0;  // 0 = dense or quant qs, 1 = quant d scales, 2 = quant m mins (Q4_K)
    PjrtDType dtype = PjrtDType::F32;
    std::vector<int64_t> dims;
    bool donatable = false;
};

struct PjrtOutputBinding {
    uint64_t tensor_id = 0;
    PjrtDType dtype = PjrtDType::F32;
    std::vector<int64_t> dims;
};

struct EmittedProgram {
    std::string cache_key;
    std::string mlir_text;
    std::vector<PjrtInputBinding> input_bindings;
    std::vector<PjrtOutputBinding> output_bindings;
};

// Emits a complete, self-contained StableHLO MLIR module ("mlir" format) and deterministic
// structural cache key for any single op or fused multi-op GGML subgraph.
EmittedProgram emit_stablehlo_program(const GraphProgramSpec& spec);

// Convenience helpers for unit tests / single-op lowering.
std::string format_mlir_tensor_type(PjrtDType dtype, const std::vector<int64_t>& dims);

}  // namespace cloudtpu
