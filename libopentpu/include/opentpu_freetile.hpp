// SPDX-License-Identifier: Apache-2.0
// Zero-dependency C++17 FreeTile Backend B descriptor & VLIW lowering contract
// for Cloud TPU v5e-1 bare-metal IMEM execution.
#ifndef OPENTPU_FREETILE_H_
#define OPENTPU_FREETILE_H_

#include "opentpu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ot {
namespace freetile {

static constexpr uint32_t kV5eMaxItersPerLaunch = 1024u;
static constexpr uint64_t kV5eQuantScaleTileDmaBytes = 1024ULL;
static constexpr uint64_t kV5eQuantConstChunkDmaBytes = 36864ULL;
static constexpr uint32_t kV5eImemBundlesBf16 = 456u;
static constexpr uint32_t kV5eImemBundlesQuantQ8 = 664u;
static constexpr uint32_t kV5eImemBundlesQuantQ4 = 512u;

inline uint64_t count_v5e_launch_chunks(uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles,
                                        uint32_t max_iters_per_launch = kV5eMaxItersPerLaunch) {
  if (m_tiles == 0u || k_tiles == 0u || n_tiles == 0u || max_iters_per_launch == 0u) return 0ULL;
  const uint32_t n_step_base =
      (k_tiles >= max_iters_per_launch) ? 1u : std::max(1u, max_iters_per_launch / k_tiles);
  const uint32_t m_step = std::min(m_tiles, n_step_base);
  const uint64_t full_m_blocks = m_tiles / m_step;
  const uint32_t rem_m = m_tiles % m_step;
  const uint64_t n_max_full = std::max(1u, n_step_base / m_step);
  const uint64_t chunks_per_full =
      (static_cast<uint64_t>(n_tiles) + n_max_full - 1ULL) / n_max_full;
  uint64_t total = full_m_blocks * chunks_per_full;
  if (rem_m > 0u) {
    const uint64_t n_max_rem = std::max(1u, n_step_base / rem_m);
    total += (static_cast<uint64_t>(n_tiles) + n_max_rem - 1ULL) / n_max_rem;
  }
  return total;
}

enum class OpKind : uint8_t {
  kMatmul = 0,
  kMatmulDequant = 1,
};

enum class DType : uint8_t {
  kBf16 = 0,
  kF32 = 1,
  kInt8 = 2,
  kQ8_0 = 3,
  kQ4_0 = 4,
  kQ4_K = 5,
  kInvalid = 0xff,
};

// Validate raw uint32_t C ABI dtype without silent narrowing to uint8_t.
inline DType dtype_from_c(uint32_t raw) {
  switch (raw) {
    case OT_DTYPE_BF16: return DType::kBf16;
    case OT_DTYPE_F32:  return DType::kF32;
    case OT_DTYPE_INT8: return DType::kInt8;
    case OT_DTYPE_Q8_0: return DType::kQ8_0;
    case OT_DTYPE_Q4_0: return DType::kQ4_0;
    case OT_DTYPE_Q4_K: return DType::kQ4_K;
    default:            return DType::kInvalid;
  }
}

inline bool is_valid_c_quant_mode(uint32_t raw) {
  return raw == OT_QUANT_NONE || raw == OT_QUANT_PER_TENSOR ||
         raw == OT_QUANT_PER_CHANNEL || raw == OT_QUANT_BLOCK_32 ||
         raw == OT_QUANT_BLOCK_256;
}

struct TileBlockSpec {
  uint32_t block_m = 128u;
  uint32_t block_k = 128u;
  uint32_t block_n = 128u;
  bool vmem_double_buffer = true;
  bool k_loop_accum_f32 = true;
};

struct QuantSpec {
  ot_quant_mode_t mode = OT_QUANT_NONE;
  uint32_t group_size = 0u;
  float per_tensor_scale = 1.0f;
  DType scale_dtype = DType::kF32;
};

struct TileKernelDescriptor {
  std::string name = "freetile_gemm_v5e";
  OpKind op_kind = OpKind::kMatmul;
  DType lhs_dtype = DType::kBf16;
  DType rhs_dtype = DType::kBf16;
  DType out_dtype = DType::kBf16;
  uint32_t m_dim = 128u;
  uint32_t k_dim = 128u;
  uint32_t n_dim = 128u;
  TileBlockSpec block_spec{};
  QuantSpec quant{};
  bool rhs_transposed = false;
  bool has_invalid_c_field = false;

  static TileKernelDescriptor from_c_desc(const ot_freetile_gemm_desc_t& c) {
    TileKernelDescriptor d;
    d.m_dim = c.m_dim;
    d.k_dim = c.k_dim;
    d.n_dim = c.n_dim;
    d.block_spec.block_m = c.tile.block_m ? c.tile.block_m : 128u;
    d.block_spec.block_k = c.tile.block_k ? c.tile.block_k : 128u;
    d.block_spec.block_n = c.tile.block_n ? c.tile.block_n : 128u;
    d.block_spec.vmem_double_buffer = (c.tile.vmem_double_buffer != 0u);
    d.block_spec.k_loop_accum_f32 = (c.tile.k_loop_accum_f32 != 0u);
    d.rhs_transposed = (c.tile.rhs_transposed != 0u);
    d.lhs_dtype = dtype_from_c(c.lhs_dtype);
    d.rhs_dtype = dtype_from_c(c.rhs_dtype);
    d.out_dtype = dtype_from_c(c.out_dtype);
    d.op_kind = (d.rhs_dtype == DType::kBf16) ? OpKind::kMatmul : OpKind::kMatmulDequant;
    const bool qm_ok = is_valid_c_quant_mode(c.quant_mode);
    d.quant.mode = qm_ok ? static_cast<ot_quant_mode_t>(c.quant_mode) : OT_QUANT_NONE;
    d.quant.group_size = c.quant_group_size;
    d.quant.per_tensor_scale = c.quant_scale;
    d.quant.scale_dtype = dtype_from_c(c.scale_dtype);
    d.has_invalid_c_field =
        !qm_ok || (c.tile.rhs_transposed > 1u) ||
        (c.struct_size != 0u && c.struct_size != sizeof(ot_freetile_gemm_desc_t));
    return d;
  }
};

struct VmemBufferPlan {
  uint32_t lhs_vmem_bytes = 0;     // Per stage (x2 when double buffered)
  uint32_t rhs_vmem_bytes = 0;     // Per stage (x2 when double buffered)
  uint32_t scale_vmem_bytes = 0;   // Scale VMEM scratch region if quantized (32 KiB cleared window)
  uint32_t dequant_vmem_bytes = 0; // BF16 dequantized RHS tile staging if quantized (32 KiB)
  uint32_t const_vmem_bytes = 0;   // Dequant Sel + IEEE-754 bias constants if quantized (36 KiB)
  uint32_t acc_vmem_bytes = 0;     // 0 on v5e (FP32 accumulators live in vector registers v48..v63)
  uint32_t out_vmem_bytes = 0;     // Output BF16 tile (32 KiB)
  uint32_t total_vmem_bytes = 0;
  uint32_t stages = 1;
};

struct LoweringPlan {
  bool valid = false;
  std::string error;
  TileKernelDescriptor desc{};
  uint32_t tiles_m = 0;
  uint32_t tiles_k = 0;
  uint32_t tiles_n = 0;
  VmemBufferPlan vmem{};
  uint32_t imem_bundles = 0;
  uint64_t hbm_bytes_moved = 0;
};

// Shared descriptor contract validator used by both plan_v5e_tile_gemm() and
// ot_matmul_bf16_ex() / ot_matmul_quant() so planner and runtime strictly agree.
inline bool validate_v5e_tile_gemm_desc(const TileKernelDescriptor& desc, std::string* error) {
  if (desc.has_invalid_c_field) {
    if (error) *error = "Invalid C descriptor field value";
    return false;
  }
  const uint32_t bm = desc.block_spec.block_m;
  const uint32_t bk = desc.block_spec.block_k;
  const uint32_t bn = desc.block_spec.block_n;
  if (desc.m_dim == 0u || desc.k_dim == 0u || desc.n_dim == 0u) {
    if (error) *error = "GEMM dimensions must be non-zero";
    return false;
  }
  if (bm != 128u || bk != 128u || bn != 128u) {
    if (error) *error = "v5e MXU native tile block must be 128x128x128";
    return false;
  }
  if ((desc.m_dim % bm) != 0u || (desc.k_dim % bk) != 0u || (desc.n_dim % bn) != 0u) {
    if (error) *error = "GEMM dimensions must be multiples of tile block (128, 128, 128)";
    return false;
  }
  if (desc.lhs_dtype != DType::kBf16 || desc.out_dtype != DType::kBf16) {
    if (error) *error = "v5e tile GEMM requires BF16 activations and outputs";
    return false;
  }
  if (desc.rhs_dtype == DType::kQ4_K || desc.quant.mode == OT_QUANT_BLOCK_256) {
    if (error) *error = "Q4_K / BLOCK_256 is reserved and not yet supported";
    return false;
  }
  if (desc.rhs_dtype == DType::kBf16) {
    if (desc.quant.mode != OT_QUANT_NONE || desc.quant.group_size != 0u) {
      if (error) *error = "BF16 tile GEMM requires OT_QUANT_NONE and quant_group_size == 0";
      return false;
    }
    return true;
  }
  if (desc.rhs_dtype != DType::kInt8 && desc.rhs_dtype != DType::kQ8_0 &&
      desc.rhs_dtype != DType::kQ4_0) {
    if (error) *error = "Unsupported RHS dtype for v5e tile GEMM";
    return false;
  }
  if (desc.quant.mode == OT_QUANT_BLOCK_32) {
    if (desc.quant.group_size != 0u && desc.quant.group_size != OT_QK8_0) {
      if (error) *error = "OT_QUANT_BLOCK_32 requires quant_group_size of 0 or 32";
      return false;
    }
  } else if (desc.quant.mode == OT_QUANT_PER_CHANNEL || desc.quant.mode == OT_QUANT_PER_TENSOR) {
    if (desc.quant.group_size != 0u) {
      if (error) *error = "Per-channel / per-tensor quant requires quant_group_size == 0";
      return false;
    }
  } else if (desc.quant.mode == OT_QUANT_NONE &&
             (desc.rhs_dtype == DType::kQ8_0 || desc.rhs_dtype == DType::kQ4_0)) {
    if (desc.quant.group_size != 0u && desc.quant.group_size != OT_QK8_0) {
      if (error) *error = "Q8_0 / Q4_0 requires quant_group_size of 0 or 32";
      return false;
    }
  } else {
    if (error) *error = "Unsupported quantization mode for RHS dtype";
    return false;
  }
  if (desc.quant.scale_dtype != DType::kBf16 && desc.quant.scale_dtype != DType::kF32) {
    if (error) *error = "Quantized GEMM scale_dtype must be OT_DTYPE_BF16 or OT_DTYPE_F32";
    return false;
  }
  return true;
}

// Validate and plan a FreeTile GEMM/Dequant-GEMM kernel for TPU v5e-1 (16 MiB VMEM).
inline LoweringPlan plan_v5e_tile_gemm(const TileKernelDescriptor& desc) {
  LoweringPlan plan;
  plan.desc = desc;
  if (!validate_v5e_tile_gemm_desc(desc, &plan.error)) {
    return plan;
  }
  const uint32_t bm = desc.block_spec.block_m;
  const uint32_t bk = desc.block_spec.block_k;
  const uint32_t bn = desc.block_spec.block_n;

  plan.tiles_m = desc.m_dim / bm;
  plan.tiles_k = desc.k_dim / bk;
  plan.tiles_n = desc.n_dim / bn;
  const bool is_quant = (desc.rhs_dtype != DType::kBf16);
  if (is_quant && plan.desc.quant.mode == OT_QUANT_NONE) {
    plan.desc.quant.mode = OT_QUANT_BLOCK_32;
    if (plan.desc.quant.group_size == 0u) plan.desc.quant.group_size = OT_QK8_0;
  }
  // Match the exact VMEM layout & DMA schedule emitted by ot_vliw.h:
  // - BF16 multi-K (tiles_k > 1): 2-stage ping-pong LHS+RHS (0x000/0x040 + 0x080/0x0c0) + OUT (0x100).
  // - BF16 single-K (tiles_k == 1): 1-stage LHS+RHS (0x000/0x040) + OUT (0x100).
  // - Quantized kernels: 1-stage LHS (0x000, 32 KiB) + packed Q (0x0c0, 8/16 KiB) +
  //   dequant BF16 RHS staging (0x040, 32 KiB) + OUT (0x100, 32 KiB) +
  //   cleared 8-VREG scale scratch window (0x140..0x178, 32 KiB) +
  //   constants (0x180 Sel 32 KiB + 0x1c0/0x1d0 bias 4 KiB = 36 KiB).
  // - FP32 accumulators (v48..v63) live in vector registers, not VMEM (acc_vmem_bytes = 0).
  const uint32_t stages = (!is_quant && plan.tiles_k > 1u) ? 2u : 1u;
  plan.desc.block_spec.vmem_double_buffer = (stages == 2u);
  // Both resident v5e IMEM kernels always accumulate across K tiles in FP32 registers (v48..v63).
  plan.desc.block_spec.k_loop_accum_f32 = true;
  plan.vmem.stages = stages;
  plan.vmem.lhs_vmem_bytes = bm * bk * 2u;
  size_t rhs_elem_num = 2u;
  size_t rhs_elem_den = 1u;
  if (desc.rhs_dtype == DType::kInt8 || desc.rhs_dtype == DType::kQ8_0) {
    rhs_elem_num = 1u;
  } else if (desc.rhs_dtype == DType::kQ4_0) {
    rhs_elem_num = 1u;
    rhs_elem_den = 2u;
  }
  plan.vmem.rhs_vmem_bytes = static_cast<uint32_t>((static_cast<uint64_t>(bk) * bn * rhs_elem_num) / rhs_elem_den);
  // Quantized emitter zeroes 8 VREGs (8 * 4 KiB = 32 KiB) at VMEM[0x140..0x178] for scale staging.
  plan.vmem.scale_vmem_bytes = is_quant ? (8u * 4096u) : 0u;
  plan.vmem.dequant_vmem_bytes = is_quant ? (bk * bn * 2u) : 0u;
  plan.vmem.const_vmem_bytes = is_quant ? static_cast<uint32_t>(kV5eQuantConstChunkDmaBytes) : 0u;
  plan.vmem.acc_vmem_bytes = 0u;
  plan.vmem.out_vmem_bytes = bm * bn * 2u;
  plan.vmem.total_vmem_bytes =
      stages * (plan.vmem.lhs_vmem_bytes + plan.vmem.rhs_vmem_bytes) +
      plan.vmem.scale_vmem_bytes + plan.vmem.dequant_vmem_bytes +
      plan.vmem.const_vmem_bytes + plan.vmem.acc_vmem_bytes + plan.vmem.out_vmem_bytes;

  // v5e TensorCore VMEM budget is 16 MiB (16,777,216 B) per core.
  if (plan.vmem.total_vmem_bytes > 16u * 1024u * 1024u) {
    plan.error = "Planned VMEM footprint exceeds 16 MiB v5e VMEM limit";
    return plan;
  }

  const uint64_t a_bytes = static_cast<uint64_t>(desc.m_dim) * desc.k_dim * 2ULL;
  const uint64_t b_bytes = (static_cast<uint64_t>(desc.k_dim) * desc.n_dim * rhs_elem_num) / rhs_elem_den;
  const uint64_t c_bytes = static_cast<uint64_t>(desc.m_dim) * desc.n_dim * 2ULL;
  // Total HBM traffic across (tiles_m x tiles_n) output tiles streaming K:
  // each output tile streams K of A and K of B (plus 1 KiB scale tile per
  // M/N/K iteration and 36 KiB constants per launch chunk when quantized),
  // then writes 128x128 C.
  uint64_t hbm_total =
      static_cast<uint64_t>(plan.tiles_n) * a_bytes +
      static_cast<uint64_t>(plan.tiles_m) * b_bytes +
      c_bytes;
  if (is_quant) {
    const uint64_t tile_iters =
        static_cast<uint64_t>(plan.tiles_m) * plan.tiles_n * plan.tiles_k;
    const uint64_t chunks =
        count_v5e_launch_chunks(plan.tiles_m, plan.tiles_k, plan.tiles_n, kV5eMaxItersPerLaunch);
    hbm_total += tile_iters * kV5eQuantScaleTileDmaBytes +
                 chunks * kV5eQuantConstChunkDmaBytes;
  }
  plan.hbm_bytes_moved = hbm_total;
  if (desc.rhs_dtype == DType::kBf16) {
    plan.imem_bundles = kV5eImemBundlesBf16;
  } else if (desc.rhs_dtype == DType::kQ4_0) {
    plan.imem_bundles = kV5eImemBundlesQuantQ4;
  } else {
    plan.imem_bundles = kV5eImemBundlesQuantQ8;
  }
  plan.valid = true;
  return plan;
}

}  // namespace freetile
}  // namespace ot

#endif  // OPENTPU_FREETILE_H_
