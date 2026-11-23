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
    d.quant.scale_dtype = static_cast<DType>(c.scale_dtype);
    return d;
  }
};

struct VmemBufferPlan {
  uint32_t lhs_vmem_bytes = 0;     // Per stage (x2 when double buffered)
  uint32_t rhs_vmem_bytes = 0;     // Per stage (x2 when double buffered)
  uint32_t scale_vmem_bytes = 0;   // Per stage if quantized (1 KiB DMA window)
  uint32_t dequant_vmem_bytes = 0; // BF16 dequantized RHS tile staging if quantized
  uint32_t const_vmem_bytes = 0;   // Dequant Sel + IEEE-754 bias constants if quantized
  uint32_t acc_vmem_bytes = 0;     // FP32 accumulator / staging
  uint32_t out_vmem_bytes = 0;     // Output BF16 tile
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
  if (desc.rhs_dtype == DType::kQ4_K || desc.quant.mode == OT_QUANT_BLOCK_256) {
    plan.error = "Q4_K / BLOCK_256 is reserved and not yet supported";
    return plan;
  }
  if (desc.rhs_dtype != DType::kBf16 && desc.rhs_dtype != DType::kInt8 &&
      desc.rhs_dtype != DType::kQ8_0 && desc.rhs_dtype != DType::kQ4_0) {
    plan.error = "Unsupported RHS dtype for v5e tile GEMM";
    return plan;
  }

  plan.tiles_m = desc.m_dim / bm;
  plan.tiles_k = desc.k_dim / bk;
  plan.tiles_n = desc.n_dim / bn;
  const bool is_quant = (desc.rhs_dtype != DType::kBf16);
  // Match the exact DMA schedule emitted by ot_vliw.h:
  // - BF16 multi-K (tiles_k > 1) always uses 2-stage ping-pong VMEM (0x000/0x040 + 0x080/0x0c0).
  // - BF16 single-K (tiles_k == 1) uses only stage 0 (0x000/0x040).
  // - Quantized IMEM kernels are single-stage (0x000 LHS, 0x0c0 Q, 0x140 scales,
  //   0x040 dequant BF16 staging, 0x180 constants).
  const uint32_t stages = (!is_quant && plan.tiles_k > 1u) ? 2u : 1u;
  plan.desc.block_spec.vmem_double_buffer = (stages == 2u);
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
  plan.vmem.scale_vmem_bytes = is_quant ? static_cast<uint32_t>(kV5eQuantScaleTileDmaBytes) : 0u;
  plan.vmem.dequant_vmem_bytes = is_quant ? (bk * bn * 2u) : 0u;
  plan.vmem.const_vmem_bytes = is_quant ? static_cast<uint32_t>(kV5eQuantConstChunkDmaBytes) : 0u;
  plan.vmem.acc_vmem_bytes = bm * bn * 4u;
  plan.vmem.out_vmem_bytes = bm * bn * 2u;
  plan.vmem.total_vmem_bytes =
      stages * (plan.vmem.lhs_vmem_bytes + plan.vmem.rhs_vmem_bytes + plan.vmem.scale_vmem_bytes) +
      plan.vmem.dequant_vmem_bytes + plan.vmem.const_vmem_bytes +
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
  plan.valid = true;
  return plan;
}

}  // namespace freetile
}  // namespace ot

#endif  // OPENTPU_FREETILE_H_
