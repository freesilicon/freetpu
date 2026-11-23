// SPDX-License-Identifier: Apache-2.0
// Zero-dependency C++17 FreeTile Backend B descriptor & VLIW lowering contract
// for Cloud TPU v5e-1 bare-metal IMEM execution.
#ifndef OPENTPU_FREETILE_H_
#define OPENTPU_FREETILE_H_

#include "opentpu.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ot {
namespace freetile {

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
};

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
    d.lhs_dtype = static_cast<DType>(c.lhs_dtype);
    d.rhs_dtype = static_cast<DType>(c.rhs_dtype);
    d.out_dtype = static_cast<DType>(c.out_dtype);
    d.op_kind = (d.rhs_dtype == DType::kBf16) ? OpKind::kMatmul : OpKind::kMatmulDequant;
    d.quant.mode = static_cast<ot_quant_mode_t>(c.quant_mode);
    d.quant.group_size = c.quant_group_size;
    d.quant.per_tensor_scale = c.quant_scale;
    return d;
  }
};

struct VmemBufferPlan {
  uint32_t lhs_vmem_bytes = 0;   // Per stage (x2 when double buffered)
  uint32_t rhs_vmem_bytes = 0;   // Per stage (x2 when double buffered)
  uint32_t scale_vmem_bytes = 0; // Per stage if quantized
  uint32_t acc_vmem_bytes = 0;   // FP32 accumulator / staging
  uint32_t out_vmem_bytes = 0;   // Output BF16 tile
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

// Validate and plan a FreeTile GEMM/Dequant-GEMM kernel for TPU v5e-1 (16 MiB VMEM).
inline LoweringPlan plan_v5e_tile_gemm(const TileKernelDescriptor& desc) {
  LoweringPlan plan;
  plan.desc = desc;
  const uint32_t bm = desc.block_spec.block_m;
  const uint32_t bk = desc.block_spec.block_k;
  const uint32_t bn = desc.block_spec.block_n;
  if (desc.m_dim == 0u || desc.k_dim == 0u || desc.n_dim == 0u) {
    plan.error = "GEMM dimensions must be non-zero";
    return plan;
  }
  if (bm != 128u || bk != 128u || bn != 128u) {
    plan.error = "v5e MXU native tile block must be 128x128x128";
    return plan;
  }
  if ((desc.m_dim % bm) != 0u || (desc.k_dim % bk) != 0u || (desc.n_dim % bn) != 0u) {
    plan.error = "GEMM dimensions must be multiples of tile block (128, 128, 128)";
    return plan;
  }
  if (desc.lhs_dtype != DType::kBf16 || desc.out_dtype != DType::kBf16) {
    plan.error = "v5e tile GEMM requires BF16 activations and outputs";
    return plan;
  }

  plan.tiles_m = desc.m_dim / bm;
  plan.tiles_k = desc.k_dim / bk;
  plan.tiles_n = desc.n_dim / bn;
  const uint32_t stages = (desc.block_spec.vmem_double_buffer && plan.tiles_k > 1u) ? 2u : 1u;
  plan.vmem.stages = stages;
  plan.vmem.lhs_vmem_bytes = bm * bk * 2u;
  size_t rhs_elem_num = 2u;
  size_t rhs_elem_den = 1u;
  if (desc.rhs_dtype == DType::kInt8 || desc.rhs_dtype == DType::kQ8_0) {
    rhs_elem_num = 1u;
  } else if (desc.rhs_dtype == DType::kQ4_0 || desc.rhs_dtype == DType::kQ4_K) {
    rhs_elem_num = 1u;
    rhs_elem_den = 2u;
  }
  plan.vmem.rhs_vmem_bytes = static_cast<uint32_t>((static_cast<uint64_t>(bk) * bn * rhs_elem_num) / rhs_elem_den);
  plan.vmem.scale_vmem_bytes = (desc.rhs_dtype == DType::kBf16) ? 0u : (bn * 4u);
  plan.vmem.acc_vmem_bytes = bm * bn * 4u;
  plan.vmem.out_vmem_bytes = bm * bn * 2u;
  plan.vmem.total_vmem_bytes =
      stages * (plan.vmem.lhs_vmem_bytes + plan.vmem.rhs_vmem_bytes + plan.vmem.scale_vmem_bytes) +
      plan.vmem.acc_vmem_bytes + plan.vmem.out_vmem_bytes;

  // v5e TensorCore VMEM budget is 16 MiB (16,777,216 B) per core.
  if (plan.vmem.total_vmem_bytes > 16u * 1024u * 1024u) {
    plan.error = "Planned VMEM footprint exceeds 16 MiB v5e VMEM limit";
    return plan;
  }

  const uint64_t a_bytes = static_cast<uint64_t>(desc.m_dim) * desc.k_dim * 2ULL;
  const uint64_t b_bytes = (static_cast<uint64_t>(desc.k_dim) * desc.n_dim * rhs_elem_num) / rhs_elem_den;
  const uint64_t c_bytes = static_cast<uint64_t>(desc.m_dim) * desc.n_dim * 2ULL;
  // Total HBM traffic across (tiles_m x tiles_n) output tiles streaming K:
  // each output tile streams K of A and K of B, then writes 128x128 C.
  plan.hbm_bytes_moved =
      static_cast<uint64_t>(plan.tiles_n) * a_bytes +
      static_cast<uint64_t>(plan.tiles_m) * b_bytes +
      c_bytes;
  plan.valid = true;
  return plan;
}

}  // namespace freetile
}  // namespace ot

#endif  // OPENTPU_FREETILE_H_
