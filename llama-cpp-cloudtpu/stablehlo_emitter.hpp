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
    Q4_0_I4,     // Single-read K-major: (qs: [B3, B2, K/32, 32, N] S4/U4 or [..., N/2] U8 bitcast, d: [B3, B2, K/32, N] F16)
    Q4_K_I4,     // Single-read K-major: (qs: [B3, B2, K/32, 32, N] S4/U4 or [..., N/2] U8 bitcast, d_eff/m_eff: [B3, B2, K/32, N] F32/BF16)
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
    PjrtDType scale_dtype = PjrtDType::INVALID;     // Optional override for Q4_K/Q6_K scale/min dtype (INVALID = default)
    PjrtDType bitcast_target = PjrtDType::INVALID;  // For Q4_*_I4 with dtype==U8: target 4-bit type (INVALID defaults to S4)

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

enum class Fp8MatmulMode {
    OFF = 0,
    AUTO = 1,
    FORCE = 2,
};

struct GraphProgramSpec {
    std::vector<TensorSpec> inputs;   // External live-in tensors
    std::vector<OpNodeSpec> ops;      // Topologically ordered ops
    std::vector<uint64_t> output_ids; // Live-out TensorSpec::ids
    int32_t spmd_chips = 1;           // Number of replica-SPMD chips (1 = single chip, 2 = tpu7x-1t, 8 = tpu7x-4t/v6e-8)
    bool bf16_matmul = true;          // Cast split-quantized weights & activations to BF16 for MXU (dense F16/F32 stay F32)
    Fp8MatmulMode fp8_matmul = Fp8MatmulMode::AUTO;  // AUTO enables f8E4M3FN dot_general when target_is_tpu7x=true
    bool target_is_tpu7x = false;     // Set when active runtime exposes Ironwood / Cloud TPU v7x hardware
    int32_t d2d_group_size = 1;       // Logical chiplets per physical package (2 on tpu7x, 1 otherwise)
    bool emit_sdy_mesh = false;       // Emit module-level Shardy `sdy.mesh` aligned to D2D pairs
    std::vector<int64_t> sdy_mesh_device_ids;  // Optional ordered logical device IDs matching DeviceAssignmentProto
    bool bf16_flash_attn = false;     // Opt-in BF16 Q/K/probs/V matmuls in FLASH_ATTN_EXT (default false: F32 parity)
    bool q4_split_dot = true;         // Q4_0/Q4_K: true = two split-X dots (zero concat); false = single concat dot
    int32_t mm_pad_m = 8;             // Pad small BF16 Q4_*_I4/Q6_K_SPLIT M < mm_pad_m up to mm_pad_m for MXU (<=1 disables)
    bool freetile_mosaic = false;     // Lower eligible quantized matmuls via FreeTile -> Mosaic tpu_custom_call
    int32_t freetile_chip_target = 0; // freetile::ChipTarget (0=V5E, 1=V6E, 2=TPU7X, 3=BUENOS_A0)
    int64_t freetile_mosaic_version = -1; // -1 = target default; >0 overrides stable_mosaic.version
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

// =========================================================================
// Continuous Batching Padded Buckets & Safe Dataflow Token-Axis Propagation
// =========================================================================

// Maximum supported batch bucket token count (prevents oversized/overflowing host padding buffers).
constexpr int64_t kMaxBatchBucketTokens = 131072;

// Overflow-checked byte count for a 4D dense tensor: dims[0]*dims[1]*dims[2]*dims[3]*elem_bytes.
// Returns false on null pointers, elem_bytes == 0, negative dims, or size_t multiplication overflow.
bool checked_tensor_nbytes(const int64_t dims[4], size_t elem_bytes, size_t* out_bytes);

// Parses CLOUDTPU_BATCH_BUCKETS (default "1,2,4,8,16,32,64,128,256,512", sorted unique
// positive ints <= kMaxBatchBucketTokens; "0" or "" disables bucketing and returns an empty vector).
std::vector<int64_t> parse_batch_buckets(const char* env_val);

// Returns smallest bucket B >= n_tokens if 1 <= n_tokens <= buckets.back(), else 0.
int64_t select_batch_bucket(int64_t n_tokens, const std::vector<int64_t>& buckets);

enum class TokenInputPadKind {
    NONE = 0,
    ZERO = 1,
    NEG_ONE = 2,
};

struct BucketAnalysisResult {
    bool success = false;
    int64_t orig_tokens = 0;
    int64_t bucket_tokens = 0;
    std::vector<int> id_token_axis;       // Dense-ID fast lookup when max TensorSpec::id is bounded
    std::vector<int> input_token_axis;    // Parallel to GraphProgramSpec::inputs (-1 or 0..3)
    std::vector<TokenInputPadKind> input_pad_kind;
    std::vector<int> op_dst_token_axis;   // Parallel to GraphProgramSpec::ops (-1 or 0..3)
    std::vector<int> output_token_axis;   // Parallel to GraphProgramSpec::output_ids (-1 or 0..3)
    std::string fail_reason;              // Populated when success == false for diagnostics
};

// Maps src_ax through a contiguous reshape from src_dims to dst_dims via unique outer/inner product
// matching where dst_dims[dst_ax] == src_dims[src_ax]. Returns -1 if missing or ambiguous.
int map_token_axis_reshape(const int64_t* src_dims,
                           size_t src_rank,
                           int src_ax,
                           const int64_t* dst_dims,
                           size_t dst_rank);

// Adjusts a SLICE descriptor along parent_token_axis (orig_tokens -> bucket_tokens).
bool bucket_view_slice_dims(const int64_t parent_dims[4],
                            int parent_token_axis,
                            int64_t orig_tokens,
                            int64_t bucket_tokens,
                            std::vector<int64_t>& reshape_dims,
                            std::vector<int64_t>& starts,
                            std::vector<int64_t>& limits);

// Propagates token_axis safely across all inputs, ops, and outputs of `spec`.
// Fails closed (success=false) on any axis conflict, ambiguity, or unsafe slice.
BucketAnalysisResult analyze_program_token_axes(const GraphProgramSpec& spec,
                                                int64_t orig_tokens,
                                                int64_t bucket_tokens);

// Rewrites `spec` dimensions in-place from orig_tokens -> bucket_tokens per `plan`.
bool apply_program_batch_bucket(GraphProgramSpec& spec, const BucketAnalysisResult& plan);

// Pads a contiguous C-order host buffer along `token_axis` from orig_dims to padded_dims.
// When `src == nullptr`, fills [0..T) with 0 and [T..B) according to `pad_kind`.
void pad_dense_host_tensor(const void* src,
                           void* dst,
                           PjrtDType dtype,
                           const int64_t orig_dims[4],
                           const int64_t padded_dims[4],
                           int token_axis,
                           TokenInputPadKind pad_kind);

// Slices logical [0..T) along `token_axis` from a padded C-order host buffer into `dst`.
void slice_dense_host_tensor(const void* src,
                             void* dst,
                             size_t elem_bytes,
                             const int64_t padded_dims[4],
                             const int64_t logical_dims[4],
                             int token_axis);

// Formats a D2D-aware Shardy `sdy.mesh` module declaration (e.g. `sdy.mesh @mesh = <["ici"=4, "d2d"=2]>` on tpu7x-4t).
std::string format_sdy_mesh_op(int32_t num_chips,
                               int32_t d2d_group_size = 1,
                               const std::vector<int64_t>& device_ids = {});

}  // namespace cloudtpu
