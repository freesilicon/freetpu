// SPDX-License-Identifier: Apache-2.0
// Clean-room Cloud TPU v5e 512-bit VLIW bundle assembler, SMEM launch ABI,
// and {1,0:T(2,128)} BF16 HBM tensor packing helpers.
#ifndef OPENTPU_VLIW_H_
#define OPENTPU_VLIW_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ot {

// Fixed BAR0 + 0x10000 ring controller queue layout & SQ opcode constants.
static constexpr uint64_t kSqRingIova      = 0x00000000ULL;
static constexpr size_t   kSqRingBytes     = 0x8000u;   // 256 x 128B descriptors
static constexpr uint64_t kCqRingIova      = 0x00008000ULL;
static constexpr size_t   kCqRingBytes     = 0x2000u;   // 256 x 32B completions
static constexpr uint64_t kStatusPageIova  = 0x0000a000ULL;
static constexpr size_t   kStatusPageBytes = 0x1000u;   // [0]=sq_head, [4]=cq_tail, [8]=active
static constexpr uint64_t kStageBaseIova   = 0x00100000ULL;
static constexpr size_t   kStageBytes      = 0x00400000u;  // 4 MiB pinned DMA staging

static constexpr uint32_t kSqOpcodeCoreCtrl = 0x03u;
static constexpr uint32_t kSqOpcodeDma      = 0x04u;
static constexpr uint32_t kSqOpcodeExec     = 0x06u;

// Default HBM IMEM/SMEM launch descriptor base (top of 16 GiB TPU aperture).
static constexpr uint64_t kDefaultLaunchDescHbmAddr = 0x3fff00000ULL;

// TPU v5e 512-bit (64-byte) VLIW instruction bundle (little-endian 16 x uint32_t).
struct VliwBundle {
  uint32_t w[16];

  static VliwBundle nop() {
    VliwBundle b{};
    b.w[0]  = 0x0f003c00u; b.w[1]  = 0x0000f000u;
    b.w[2]  = 0x3c00000fu; b.w[3]  = 0x00780000u;
    b.w[4]  = 0x0000f000u; b.w[5]  = 0x00003c00u;
    b.w[6]  = 0x0000f000u; b.w[7]  = 0x0003c000u;
    b.w[8]  = 0x000f0000u; b.w[9]  = 0x003c0000u;
    b.w[10] = 0x00000000u; b.w[11] = 0x00000000u;
    b.w[12] = 0x00000000u; b.w[13] = 0x00000000u;
    b.w[14] = 0x0f000000u; b.w[15] = 0x55780000u;
    return b;
  }

  void set_s0(uint32_t lo, uint32_t hi) { w[14] = lo; w[15] = hi; }
  void set_s1_imm(uint32_t imm_w13, uint32_t ctrl_w12 = 0) {
    if (ctrl_w12 != 0) w[12] = ctrl_w12;
    w[13] = imm_w13;
  }
  void set_dma_sync(uint32_t w2_op, uint32_t w3_val = 0x00780000u) {
    w[2] = w2_op;
    w[3] = w3_val;
  }
  void set_v0(uint32_t w4_val, uint32_t w5_val = 0x00003c00u) {
    w[4] = w4_val;
    w[5] = w5_val;
  }
  void set_v1(uint32_t w7_val, uint32_t w8_val, uint32_t w9_val) {
    w[7] = w7_val;
    w[8] = w8_val;
    w[9] = w9_val;
  }
  void set_mxu_push_weight(int bank, int parity) {
    w[0] = 0xaf003c00u;
    w[1] = 0x7018e709u | (static_cast<uint32_t>(bank & 1) << 24) |
           (static_cast<uint32_t>(parity & 1) << 4);
  }
  void set_mxu_push_act(uint32_t slot, uint32_t vreg_enc) {
    w[0] = 0xaf003c00u;
    w[1] = ((slot & 0xfu) << 24) | 0x0008e008u | ((vreg_enc & 0xfu) << 4);
  }
};

// Clean-room 512-byte HBM launch descriptor (uploaded via op=0x04 with
// dst_flags=0x40000000, w28=0x0001fb10, w29=0x01000000 before op=0x06 EXEC).
// Verified on TPU v5e-1 with synthetic zeroed timestamp/fingerprint slots.
struct OtV5eLaunchDesc {
  uint8_t bytes[512];

  static OtV5eLaunchDesc build(uint64_t lhs_hbm, uint64_t rhs_hbm, uint64_t out_hbm,
                               uint64_t prog_hbm, uint32_t exec_seq) {
    OtV5eLaunchDesc d{};
    std::memset(d.bytes, 0, sizeof(d.bytes));
    auto put32 = [&](size_t off, uint32_t v) { std::memcpy(d.bytes + off, &v, 4); };
    put32(0x170u, static_cast<uint32_t>(rhs_hbm >> 9));
    put32(0x174u, static_cast<uint32_t>(lhs_hbm >> 9));
    put32(0x17cu, 0xffffffffu);
    put32(0x190u, static_cast<uint32_t>(out_hbm >> 9));
    put32(0x1a0u, 0x20u);
    put32(0x1a4u, static_cast<uint32_t>(prog_hbm >> 9));
    put32(0x1a8u, 1u);
    put32(0x1acu, 1u);
    put32(0x1e4u, static_cast<uint32_t>(prog_hbm >> 9));
    put32(0x1e8u, exec_seq);
    return d;
  }
};

// Pack row-major 128x128 BF16 matrix into TPU v5e {1,0:T(2,128)} HBM layout:
//   tiled_u16_idx(r, c) = (r >> 1) * 256 + c * 2 + (r & 1).
inline void tile_pack_128x128_bf16(const uint16_t* row_major, uint16_t* tiled) {
  for (uint32_t r = 0; r < 128u; ++r) {
    for (uint32_t c = 0; c < 128u; ++c) {
      tiled[((r >> 1) * 256u) + (c * 2u) + (r & 1u)] = row_major[(r * 128u) + c];
    }
  }
}

inline void tile_unpack_128x128_bf16(const uint16_t* tiled, uint16_t* row_major) {
  for (uint32_t r = 0; r < 128u; ++r) {
    for (uint32_t c = 0; c < 128u; ++c) {
      row_major[(r * 128u) + c] = tiled[((r >> 1) * 256u) + (c * 2u) + (r & 1u)];
    }
  }
}

// Emit the 58-bundle 128x128 BF16 MXU matmul kernel body (HBM->VMEM DMA for LHS/RHS,
// 16x8-row MXU weight pushes, 4x4KiB activation loads, 8 MXU matmul-accumulate steps,
// 8 VSTORE accumulator flushes, VMEM->HBM output DMA, sync restore, and scalar return).
// Byte-exact parity verified on live TPU v5e-1 silicon against general BF16 A @ B.
inline std::vector<VliwBundle> emit_v5e_matmul_128x128_body() {
  std::vector<VliwBundle> p;
  p.reserve(58);
  auto push = [&](const VliwBundle& b) { p.push_back(b); };

  // 1) DMA setup & RHS HBM->VMEM (32 KiB)
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0c070fu); b.set_s1_imm(0x00100000u, 0xd8000000u); b.set_s0(0x6e01b030u, 0x5572e1d1u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xae01beb8u, 0x5572ecd1u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xef000000u, 0x55700dd1u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[9]=0x783c0000u; b.w[10]=0x000001ceu; b.set_s1_imm(0x01000000u, 0xd4000000u); b.set_s0(0x0011b0c0u, 0x5570eb80u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a8ca30fu); b.set_s1_imm(0x01000000u, 0xd4000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca10fu, 0x18700810u); b.set_v0(0x0000e000u); b.w[11]=0x01c00000u; b.set_s1_imm(0xff000000u, 0xd4001e00u); b.set_s0(0xcf000003u, 0x5572e2d0u); push(b); }

  // 2) LHS DMA setup interleaved with 16x 8-row MXU weight pushes
  { VliwBundle b = VliwBundle::nop(); b.set_mxu_push_weight(0, 0); b.set_dma_sync(0x3808030eu, 0x08701808u); b.set_v0(0x0000e020u); b.set_s1_imm(0x001a0001u, 0x80001600u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_mxu_push_weight(1, 1); b.set_dma_sync(0x380e030eu, 0x08703008u); b.set_v0(0x0000e050u); b.set_s1_imm(0x00140001u, 0x20001000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3898020fu, 0x00724804u); b.set_v0(0x0000e480u); b.set_s1_imm(0x00040000u, 0x20000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x389a030fu, 0x08725808u); b.set_v0(0x0000e4a0u); b.set_s1_imm(0x00080000u, 0xc0000600u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_mxu_push_weight(0, 0); b.set_dma_sync(0x3c00000eu, 0x1072780cu); b.set_v0(0x2000e4e0u); b.w[9]=0x003c0020u; b.set_s1_imm(0x00200000u, 0xa0000e00u); b.set_s0(0x0f000000u, 0x55700d80u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_mxu_push_weight(1, 1); b.set_dma_sync(0x3c00000eu); b.set_v0(0x2000f000u); b.w[9]=0x003c0020u; b.set_s0(0x6f000000u, 0x5572e0d0u); push(b); }
  for (int t = 2; t <= 7; ++t) {
    uint32_t voff = static_cast<uint32_t>(t * 0x20);
    for (int bank = 0; bank < 2; ++bank) {
      VliwBundle b = VliwBundle::nop();
      b.set_mxu_push_weight(bank, bank);
      b.set_dma_sync(0x3c00000eu);
      b.set_v0((static_cast<uint32_t>(t * 2) << 28) | 0x0000f000u);
      b.w[9] = 0x003c0000u | voff;
      push(b);
    }
  }

  // 3) Load 4 x 4 KiB LHS activation strips into MXU staging registers
  static constexpr uint32_t kActSlots[4] = {4u, 5u, 2u, 3u};
  for (int i = 0; i < 4; ++i) {
    VliwBundle b = VliwBundle::nop();
    b.set_mxu_push_act(kActSlots[i], kActSlots[i]);
    b.set_dma_sync(0x3c00000eu);
    b.set_v0((static_cast<uint32_t>(i) << 30) | 0x0000f000u, 0x00003c01u);
    b.w[9] = 0x003c0120u + static_cast<uint32_t>(i) * 0x40u;
    push(b);
  }

  // 4) Execute 8 MXU multiply-accumulate passes and VSTORE tiles to VMEM
  static constexpr uint32_t kMxuMatmulOps[8] = {
      0x0e443951u, 0x0e64b9d3u, 0x0e453955u, 0x0e6639d9u,
      0x0e47395du, 0x0e67b9dfu, 0x0e483961u, 0x0e6939e5u,
  };
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[0]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[1]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[2]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v1(0x155fc000u, 0x516e3a88u, 0x0038ea00u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0xc010f000u, 0x00003812u); b.set_s1_imm(0x00200000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0xe010f000u, 0x00003812u); b.set_s1_imm(0x00240000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[3]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[4]; b.set_v1(0x196fc000u, 0x61ae3a98u, 0x0038ea40u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0x4010f000u, 0x00003813u); b.set_s1_imm(0x00280000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0x6010f000u, 0x00003813u); b.set_s1_imm(0x002c0000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[5]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[6]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v1(0x218fc000u, 0x822e3ae8u, 0x0038eb80u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0x4010f000u, 0x00003814u); b.set_s1_imm(0x00220000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0x6010f000u, 0x00003814u); b.set_s1_imm(0x00260000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.w[0] = kMxuMatmulOps[7]; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v1(0x259fc000u, 0x926e3af8u, 0x0038ebc0u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0xc010f000u, 0x00003814u); b.set_s1_imm(0x002a0000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_v0(0xe010f000u, 0x00003814u); b.set_s1_imm(0x002e0000u); push(b); }

  // 5) Trigger VMEM->HBM output DMA (32 KiB), wait sync, restore flags, return
  { VliwBundle b = VliwBundle::nop(); b.w[9]=0x783c0000u; b.w[10]=0x000001ceu; b.set_s1_imm(0x01000000u, 0xd8000000u); b.set_s0(0x0800d0c0u, 0x5570e380u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a8ca30fu); b.set_s1_imm(0x01000000u, 0xd8000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca10fu); b.set_s1_imm(0xff000000u, 0xd8000000u); b.set_s0(0x0f000003u, 0x55780000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca70fu); b.set_s1_imm(0x00004000u, 0xd4000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca70fu); b.set_s1_imm(0x00004000u, 0xd8000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca60fu); b.set_s1_imm(0x00008240u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0ca60fu); b.set_s1_imm(0x3fffc27fu, 0xfc000000u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(0x3fffc1ffu, 0xfc000000u); b.set_s0(0xef000000u, 0x55700db1u); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a00250fu); b.w[10]=0x000001e0u; push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a00030fu); push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x0f000000u, 0x55700900u); push(b); }

  return p;
}

}  // namespace ot

#endif  // OPENTPU_VLIW_H_
