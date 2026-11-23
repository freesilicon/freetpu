// SPDX-License-Identifier: Apache-2.0
// Clean-room Cloud TPU v5e 512-bit VLIW bundle assembler, SMEM launch ABI,
// and {1,0:T(2,128)} BF16 HBM tensor packing helpers.
#ifndef OPENTPU_VLIW_H_
#define OPENTPU_VLIW_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "opentpu_freetile.hpp"

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

// Default HBM launch descriptor slot-0 base (top of 16 GiB TPU aperture).
static constexpr uint64_t kDefaultLaunchDescHbmAddr = 0x3fff00000ULL;

// Verified v5e direct-in-IMEM resident loop & sync/doorbell constants.
static constexpr size_t   kImemDmaAlignBytes   = 512u;
static constexpr size_t   kResidentImemBundles = 120u;
static constexpr size_t   kResidentImemBytes   = kResidentImemBundles * 64u;  // 7,680 B (15 x 512 B)
static constexpr uint32_t kCoreCtrlStart       = 0x121u;
static constexpr uint32_t kExecOpW8            = 0x41u;
static constexpr uint32_t kExecOpW10           = 0x010101f9u;
static constexpr uint32_t kSyncFlagLhsDma      = 0x33u;
static constexpr uint32_t kSyncFlagExecDone    = 0x1f9u;
static constexpr uint32_t kSyncFlagDoorbell    = 0x1fbu;
static constexpr uint32_t kSyncFlagLdescDma    = 0x1ffu;
static constexpr uint32_t kLdescDmaW28Doorbell = 0x0001fb10u;  // flag 0x1fb, +16 units (512 B / 32 B)
static constexpr uint32_t kLdescDmaW29         = 0x01000000u;
static constexpr uint32_t kImemDmaDstFlagsHi   = 0x10000000u;
static constexpr uint32_t kImemDmaW28          = 0x130u;
static constexpr uint32_t kImemDmaW29          = 0x04000000u;

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

  // Sync flag immediate primitives (9-bit flag_id 0..511, units32 = bytes / 32).
  static VliwBundle make_sync_wait_imm(uint32_t flag_id, uint32_t bytes) {
    VliwBundle b = nop();
    const uint32_t units32 = bytes / 32u;
    b.set_dma_sync(0x3a8ca30fu);
    b.w[12] = (flag_id & 0x3fu) << 26;
    b.w[13] = (units32 << 14) | ((flag_id >> 6) & 0x7u);
    return b;
  }

  static VliwBundle make_sync_sub_imm(uint32_t flag_id, uint32_t bytes) {
    VliwBundle b = nop();
    const uint32_t units32 = bytes / 32u;
    const uint32_t neg_units = (0u - units32) & 0x3ffffu;
    b.set_dma_sync(0x3a0ca10fu);
    b.w[12] = (flag_id & 0x3fu) << 26;
    b.w[13] = (neg_units << 14) | ((flag_id >> 6) & 0x7u);
    b.w[14] = 0x0f000003u;
    return b;
  }

  static VliwBundle make_sync_clear(uint32_t flag_id) {
    VliwBundle b = nop();
    b.set_dma_sync(0x3a0ca70fu);
    b.w[12] = (flag_id & 0x3fu) << 26;
    b.w[13] = 0x00004000u | ((flag_id >> 6) & 0x7u);
    return b;
  }

  // Load 32-bit word from SMEM launch descriptor (indexed relative to end +0x200).
  static VliwBundle make_smem_load_ldesc(uint32_t dst_sreg, uint32_t ldesc_byte_off) {
    VliwBundle b = nop();
    const int32_t neg_dwords = (static_cast<int32_t>(ldesc_byte_off) - 0x200) / 4;
    b.w[13] = static_cast<uint32_t>(neg_dwords) << 14;
    b.w[14] = 0x0e041000u | ((dst_sreg & 0x1fu) << 2);
    return b;
  }

  // Scalar memory/DMA fence.
  static VliwBundle make_fence() {
    VliwBundle b = nop();
    b.set_s0(0x0f000000u, 0x55700900u);
    return b;
  }

  // Unconditional absolute IMEM branch (caller must follow with 4 delay-slot bundles).
  static VliwBundle make_br_abs(uint32_t target_pc) {
    VliwBundle b = nop();
    b.w[13] = target_pc << 14;
    b.set_s0(0x0f000000u, 0x55700400u);
    return b;
  }

  // Relative IMEM branch: rel_pc = target_pc - branch_pc (followed by 4 delay slots).
  // pred5: 0x0e=always, p_reg=@pK, 0x10|p_reg=@!pK.
  static VliwBundle make_br_rel(int32_t rel_pc, uint32_t pred5 = 0x0eu) {
    VliwBundle b = nop();
    b.w[13] = (static_cast<uint32_t>(rel_pc) & 0x3ffffu) << 14;
    b.set_s0((rel_pc < 0) ? 0x0f000003u : 0x0f000000u,
             0x55000500u | ((pred5 & 0x1fu) << 19));
    return b;
  }

  // Scalar S0 opcodes & inline constants.
  static constexpr uint32_t kS0PredAlways = 0x0eu;
  static constexpr uint32_t kS0OpAdd      = 0x0au;
  static constexpr uint32_t kS0OpSub      = 0x0cu;
  static constexpr uint32_t kS0OpAnd      = 0x0eu;
  static constexpr uint32_t kS0OpOr       = 0x0fu;
  static constexpr uint32_t kS0OpMul      = 0x14u;
  static constexpr uint32_t kS0OpShll     = 0x17u;
  static constexpr uint32_t kS0OpShrl     = 0x18u;
  static constexpr uint32_t kS0OpEq       = 0x1eu;
  static constexpr uint32_t kS0OpNe       = 0x1fu;
  static constexpr uint32_t kS0OpGe       = 0x21u;
  static constexpr uint32_t kS0OpLt       = 0x22u;
  static constexpr uint32_t kSyImm0       = 32u;
  static constexpr uint32_t kSyConst0     = 48u;
  static constexpr uint32_t kSyConst1     = 49u;
  static constexpr uint32_t kSyConst4     = 52u;

  void set_s0_simm(uint32_t sd, uint32_t imm18) {
    w[13] = (imm18 & 0x3ffffu) << 14;
    w[14] = 0x0f000000u | ((sd & 7u) << 29);
    w[15] = 0x55700d80u | ((sd >> 3) & 3u);
  }

  void set_s0_alu(uint32_t op6, uint32_t sd, uint32_t sx5, uint32_t sy6,
                  uint32_t pred5 = kS0PredAlways) {
    w[14] = 0x0f000000u | ((sd & 7u) << 29);
    w[15] = 0x55000000u | ((pred5 & 0x1fu) << 19) | ((op6 & 0x3fu) << 13) |
            ((sx5 & 0x1fu) << 8) | ((sy6 & 0x3fu) << 2) | ((sd >> 3) & 3u);
  }

  void set_s0_alu_imm(uint32_t op6, uint32_t sd, uint32_t sx5, uint32_t imm18,
                      uint32_t pred5 = kS0PredAlways) {
    w[13] = (imm18 & 0x3ffffu) << 14;
    set_s0_alu(op6, sd, sx5, kSyImm0, pred5);
  }

  // Scalar simple DMA (addresses in 32-byte units: vmem_dma = 512B<<4, hbm_dma = HBM>>5).
  static VliwBundle make_dma_hbm_to_vmem_imm(uint32_t s_vmem_dst, uint32_t s_hbm_src,
                                           uint32_t flag_id, uint32_t units32) {
    VliwBundle b = nop();
    b.w[9]  = 0x783c0000u;
    b.w[10] = 0x000001ceu;
    b.w[12] = (flag_id & 0x3fu) << 26;
    b.w[13] = ((units32 & 0x3ffffu) << 14) | ((flag_id >> 6) & 0x7u);
    b.set_s0(0x001010c0u | ((s_vmem_dst & 0x1fu) << 13),
             0x5570e080u | ((s_hbm_src & 0x1fu) << 8));
    return b;
  }

  static VliwBundle make_dma_vmem_to_hbm_imm(uint32_t s_hbm_dst, uint32_t s_vmem_src,
                                           uint32_t flag_id, uint32_t units32) {
    VliwBundle b = nop();
    b.w[9]  = 0x783c0000u;
    b.w[10] = 0x000001ceu;
    b.w[12] = (flag_id & 0x3fu) << 26;
    b.w[13] = ((units32 & 0x3ffffu) << 14) | ((flag_id >> 6) & 0x7u);
    b.set_s0(0x080010c0u | ((s_hbm_dst & 0x1fu) << 13),
             0x5570e080u | ((s_vmem_src & 0x1fu) << 8));
    return b;
  }

  // VMEM vector load/store in 512-byte units (1 VREG = 8 units = 4 KiB).
  void set_vld(uint32_t vd, uint32_t vmem_off_512b) {
    w[3] = (vmem_off_512b == 0u) ? 0x00780000u : 0x08780000u;
    w[4] = 0x0000e000u | ((vd & 0x3fu) << 4);
    if (vmem_off_512b != 0u) w[13] = (vmem_off_512b & 0x3ffffu) << 14;
  }

  static VliwBundle make_vld(uint32_t vd, uint32_t vmem_off_512b) {
    VliwBundle b = nop();
    b.set_vld(vd, vmem_off_512b);
    return b;
  }

  static VliwBundle make_vst(uint32_t vmem_off_512b, uint32_t vs) {
    VliwBundle b = nop();
    b.set_v0(((vs & 7u) << 29) | 0x0010f000u, 0x00003810u | ((vs >> 3) & 0x7u));
    if (vmem_off_512b != 0u) b.w[13] = (vmem_off_512b & 0x3ffffu) << 14;
    return b;
  }

  // VPU slot V1 operations (op_w9 encodings verified on v5e-1).
  static constexpr uint32_t kV1OpCvtS32F32    = 0x0005u;
  static constexpr uint32_t kV1OpUnpackBf16Lo = 0x0804u;
  static constexpr uint32_t kV1OpUnpackBf16Hi = 0x0805u;
  static constexpr uint32_t kV1OpUnpackS8Base = 0x080eu;  // + byte_lane 0..3
  static constexpr uint32_t kV1OpAddS32Imm    = 0x180du;
  static constexpr uint32_t kV1OpAndU32Imm    = 0x3007u;
  static constexpr uint32_t kV1OpShllU32Imm   = 0x4807u;
  static constexpr uint32_t kV1OpShrlU32Imm   = 0x5007u;
  static constexpr uint32_t kV1OpShraS32Imm   = 0x5807u;
  static constexpr uint32_t kV1OpAddF32       = 0x6000u;
  static constexpr uint32_t kV1OpMulF32       = 0x7000u;
  static constexpr uint32_t kV1OpPackF32Bf16  = 0xe800u;

  void set_v1_alu(uint32_t op_w9, uint32_t vd, uint32_t vx, uint32_t vy) {
    w[8] = ((vx & 0x3fu) << 26) | ((vd & 0x3fu) << 20) | 0x000f0000u;
    w[9] = 0x00380000u | op_w9 | ((vy & 0x3fu) << 5);
  }

  static VliwBundle make_v1_alu(uint32_t op_w9, uint32_t vd, uint32_t vx, uint32_t vy) {
    VliwBundle b = nop();
    b.set_v1_alu(op_w9, vd, vx, vy);
    return b;
  }

  static VliwBundle make_v1_alu_imm(uint32_t op_w9, uint32_t vd, uint32_t vy, int32_t imm18) {
    VliwBundle b = nop();
    b.set_v1_alu(op_w9, vd, 0u, vy);
    b.w[13] = (static_cast<uint32_t>(imm18) & 0x3ffffu) << 14;
    if (imm18 < 0) b.w[14] = 0x0f000003u;
    return b;
  }

  static VliwBundle make_v1_zero(uint32_t vd) {
    return make_v1_alu_imm(kV1OpAndU32Imm, vd, vd, 0);
  }

  void set_mxu_push_weight_vreg(int bank, int parity, uint32_t vreg) {
    set_mxu_push_weight(bank, parity);
    w[2] = 0x3c00000eu;
    set_v0(((vreg & 7u) << 29) | 0x0000f000u, 0x00003c00u | ((vreg >> 3) & 0x7u));
    w[9] = 0x003c0000u | ((vreg & 0x3fu) << 5);
  }

  void set_mxu_push_act_vregs(uint32_t slot, uint32_t vA_even, uint32_t vA_odd) {
    set_mxu_push_act(slot, slot);
    w[2] = 0x3c00000eu;
    set_v0(((vA_even & 7u) << 29) | 0x0000f000u, 0x00003c00u | ((vA_even >> 3) & 0x7u));
    w[9] = 0x003c0000u | ((vA_odd & 0x3fu) << 5);
  }

  static VliwBundle make_mxu_pop(int mrf_pair_hi, uint32_t vd0, uint32_t vd1) {
    VliwBundle b = nop();
    const uint32_t base = mrf_pair_hi ? 0x0e6039c0u : 0x0e403940u;
    b.w[0] = base | ((vd0 & 0x3fu) << 14) | (vd1 & 0x3fu);
    return b;
  }
};

// Clean-room 512-byte HBM launch descriptor (uploaded via op=0x04 with
// dst_flags=0x40000000, w28=0x0001fb10, w29=0x01000000 before op=0x06 EXEC).
// Verified on TPU v5e-1 with synthetic zeroed timestamp/fingerprint slots.
struct OtV5eLaunchDesc {
  uint8_t bytes[512];

  static OtV5eLaunchDesc build(uint64_t lhs_hbm, uint64_t rhs_hbm, uint64_t out_hbm,
                               uint64_t prog_hbm, uint32_t exec_seq) {
    return build_ex(lhs_hbm, rhs_hbm, out_hbm, prog_hbm, exec_seq,
                    /*m_tiles=*/1u, /*k_tiles=*/1u, /*n_tiles=*/1u,
                    /*quant_mode=*/0u, /*scales_hbm=*/0ULL, /*const_hbm=*/0ULL);
  }

  static OtV5eLaunchDesc build_ex(uint64_t lhs_hbm, uint64_t rhs_hbm, uint64_t out_hbm,
                                  uint64_t prog_hbm, uint32_t exec_seq,
                                  uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles,
                                  uint32_t quant_mode, uint64_t scales_hbm,
                                  uint64_t const_hbm, uint32_t n_stride_tiles = 0u) {
    OtV5eLaunchDesc d{};
    std::memset(d.bytes, 0, sizeof(d.bytes));
    auto put32 = [&](size_t off, uint32_t v) { std::memcpy(d.bytes + off, &v, 4); };
    put32(0x170u, static_cast<uint32_t>(rhs_hbm >> 9));
    put32(0x174u, static_cast<uint32_t>(lhs_hbm >> 9));
    put32(0x178u, m_tiles);
    put32(0x17cu, 0xffffffffu);
    put32(0x180u, k_tiles);
    put32(0x184u, n_tiles);
    put32(0x188u, quant_mode);
    put32(0x18cu, static_cast<uint32_t>(scales_hbm >> 9));
    put32(0x190u, static_cast<uint32_t>(out_hbm >> 9));
    put32(0x194u, static_cast<uint32_t>(const_hbm >> 9));
    put32(0x198u, n_stride_tiles);
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

// Emit the 120-bundle (7,680 B, 15 x 512 B) clean-room direct-in-IMEM resident loop
// for Cloud TPU v5e-1. Uploaded once directly from host staging memory to IMEM PC=0
// (zero prog_hbm bounce/reload) and started with op=0x03 (d[8]=0x121).
//
// Stages:
//   PC   0.. 16: Signal SMEM[+0x1e8] (exec_seq) -> sync[0x1f9] (0 on cold boot).
//   PC  17.. 20: Wait & consume host doorbell on sync[0x1fb] (>= 16, -= 16).
//   PC  21.. 27: Fetch 512 B OtV5eLaunchDesc from HBM 0x3fff00000 -> SMEM + sync[0x1ff] + fence.
//   PC  28.. 50: Load (out, lhs, rhs) >> 9 into (s2, s18, s1), issue unconditional 32 KiB LHS
//                HBM->VMEM DMA, wait/sub/clear sync[0x33], zero s0.
//   PC  51..108: Embedded 58-bundle 128x128 BF16 MXU matmul body.
//   PC 109..119: NOP + absolute branch to PC=0 + 4 delay-slot NOPs + 5 pad NOPs.
inline std::vector<VliwBundle> emit_v5e_resident_matmul_imem() {
  std::vector<VliwBundle> p;
  p.reserve(kResidentImemBundles);
  auto push = [&](const VliwBundle& b) { p.push_back(b); };
  auto push_nops = [&](size_t count) {
    for (size_t i = 0; i < count; ++i) p.push_back(VliwBundle::nop());
  };

  // Stage 1 (PC 0..16): Read SMEM[+0x1e8] (exec_seq) and write to sync[0x1f9].
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3b00060fu); b.set_s0(0x0f000000u, 0x55700dc4u); push(b); }  // PC 0
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x2f000000u, 0x55701100u); push(b); }                              // PC 1
  push_nops(1);                                                                                                    // PC 2
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a00000fu); b.set_s1_imm(0xfffe8007u, 0xec000000u); b.set_s0(0x8e04100cu, 0x55714184u); push(b); }  // PC 3
  push_nops(1);                                                                                                    // PC 4
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3b0e060fu); b.set_s1_imm(0x3fffc1ffu, 0xfc007dc0u); b.set_s0(0xcf000000u, 0x55700db0u); push(b); }  // PC 5
  push_nops(1);                                                                                                    // PC 6
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xef000000u, 0x5571c60cu); push(b); }                              // PC 7
  push_nops(5);                                                                                                    // PC 8..12
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x6f000000u, 0x5571ea1du); push(b); }                              // PC 13
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0a200fu); b.w[10] = 0x00000160u; b.set_s1_imm(kSyncFlagExecDone << 14); push(b); }  // PC 14
  push_nops(2);                                                                                                    // PC 15..16

  // Stage 2 (PC 17..20): Block until host doorbell sync[0x1fb] >= 16, then subtract 16.
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x8f000000u, 0x55701101u); push(b); }                              // PC 17
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a82a30fu); b.w[10] = 0x00000080u; b.set_s1_imm(0x00040000u); push(b); }  // PC 18
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a02a10fu); b.w[10] = 0x00000080u; b.set_s1_imm(0xfffc0000u); b.set_s0(0x0f000003u, 0x55780000u); push(b); }  // PC 19
  push_nops(1);                                                                                                    // PC 20

  // Stage 3 (PC 21..27): Fetch 512 B OtV5eLaunchDesc from HBM 0x3fff00000 -> SMEM + wait/sub sync[0x1ff] + fence.
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(0x3e000007u, 0xfcffe000u); b.set_s0(0xae01b13cu, 0x55700db1u); push(b); }  // PC 21
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xcf000000u, 0x55714d31u); push(b); }                              // PC 22
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x0f000000u, 0x5572eed2u); push(b); }                              // PC 23
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(kSyncFlagLdescDma << 14); b.set_s0(0x0211f000u, 0x5570d0e4u); push(b); }  // PC 24
  push(VliwBundle::make_sync_wait_imm(kSyncFlagLdescDma, 512u));                                                  // PC 25
  push(VliwBundle::make_sync_sub_imm(kSyncFlagLdescDma, 512u));                                                   // PC 26
  push(VliwBundle::make_fence());                                                                                  // PC 27

  // Stage 4 (PC 28..50): Load (out, lhs, rhs) >> 9, unconditional 32 KiB LHS HBM->VMEM DMA, sync[0x33], s0=0.
  push(VliwBundle::make_smem_load_ldesc(2u,  0x190u));                                                            // PC 28 (s2  = out_hbm >> 9)
  push(VliwBundle::make_smem_load_ldesc(18u, 0x174u));                                                            // PC 29 (s18 = lhs_hbm >> 9)
  push(VliwBundle::make_smem_load_ldesc(1u,  0x170u));                                                            // PC 30 (s1  = rhs_hbm >> 9)
  push_nops(6);                                                                                                    // PC 31..36
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0a210fu); b.w[10] = 0x00000320u; b.set_s1_imm(kSyncFlagLhsDma << 14); b.set_s0(0x4e01bee8u, 0x55780000u); push(b); }  // PC 37
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x6e01b874u, 0x5572f2d3u); push(b); }                              // PC 38
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x8e5fba78u, 0x5572fad3u); push(b); }                              // PC 39
  push_nops(5);                                                                                                    // PC 40..44
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(kSyncFlagLhsDma << 14); b.set_s0(0x0013d000u, 0x5570db72u); push(b); }  // PC 45
  push(VliwBundle::make_sync_wait_imm(kSyncFlagLhsDma, 32768u));                                                  // PC 46
  push(VliwBundle::make_sync_sub_imm(kSyncFlagLhsDma, 32768u));                                                   // PC 47
  push(VliwBundle::make_sync_clear(kSyncFlagLhsDma));                                                             // PC 48
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x0f000000u, 0x55700dc0u); push(b); }                              // PC 49 (s0 = 0)
  push_nops(1);                                                                                                    // PC 50

  // Stage 5 (PC 51..108): 58-bundle 128x128 BF16 MXU matmul kernel body.
  const std::vector<VliwBundle> body = emit_v5e_matmul_128x128_body();
  p.insert(p.end(), body.begin(), body.end());

  // Stage 6 (PC 109..119): Loop back to PC=0 (+ 4 delay-slot NOPs + 5 pad NOPs -> 120 bundles = 7,680 B).
  push_nops(1);                                                                                                    // PC 109
  push(VliwBundle::make_br_abs(0u));                                                                              // PC 110
  push_nops(4);                                                                                                    // PC 111..114 (delay slots)
  push_nops(5);                                                                                                    // PC 115..119 (512B alignment pad)

  return p;
}

// Kernel mode selector passed via OtV5eLaunchDesc[+0x188] (SMEM quant_mode word).
enum class QuantKernelMode : uint32_t {
  kBf16           = 0u,
  kQ8_0           = 1u,
  kQ4_0           = 2u,
  kInt8PerChannel = 3u,
  kInt4PerChannel = 4u,
};

// Pack row-major 128x128 signed int8 tile into TPU v5e {1,0:T(8,128)(4,1)} HBM
// layout (16 KiB = 4,096 uint32 words = 4 VREGs of 32 K-rows x 128 N-cols).
inline void tile_pack_128x128_i8(const int8_t* row_major, uint32_t* tiled_u32) {
  std::memset(tiled_u32, 0, 128u * 128u);
  for (uint32_t r = 0; r < 128u; ++r) {
    const uint32_t chunk = r >> 5;
    const uint32_t byte_idx = (r >> 3) & 3u;
    const uint32_t sublane = r & 7u;
    const uint32_t shift = byte_idx * 8u;
    for (uint32_t c = 0; c < 128u; ++c) {
      const uint32_t u = static_cast<uint8_t>(row_major[r * 128u + c]);
      const uint32_t idx = chunk * 1024u + sublane * 128u + c;
      tiled_u32[idx] |= (u << shift);
    }
  }
}

// Pack row-major 128x128 signed int4 tile (each element clamped to [-8, 7],
// encoded as unsigned nibble q + 8 in [0, 15]) into TPU v5e {1,0:T(8,128)(4,1)}
// 2x4-bit HBM layout (8 KiB = 2,048 uint32 words = 2 VREGs of 64 K-rows).
inline void tile_pack_128x128_q4(const int8_t* row_major_q4, uint32_t* tiled_u32) {
  std::memset(tiled_u32, 0, (128u * 128u) / 2u);
  for (uint32_t r = 0; r < 128u; ++r) {
    const uint32_t chunk4 = r >> 6;
    const uint32_t hi_nib = (r >> 5) & 1u;
    const uint32_t byte_idx = (r >> 3) & 3u;
    const uint32_t sublane = r & 7u;
    const uint32_t shift = byte_idx * 8u + hi_nib * 4u;
    for (uint32_t c = 0; c < 128u; ++c) {
      int32_t v = static_cast<int32_t>(row_major_q4[r * 128u + c]);
      if (v < -8) v = -8;
      if (v > 7) v = 7;
      const uint32_t nib = static_cast<uint32_t>(v + 8) & 0xfu;
      const uint32_t idx = chunk4 * 1024u + sublane * 128u + c;
      tiled_u32[idx] |= (nib << shift);
    }
  }
}

// Pack 4 K-blocks x 128 BF16 scales (512 BF16 = 1,024 B) for one 128x128 tile
// into chunk-0 rows 0..3 of {1,0:T(2,128)} for MXU selector-matrix broadcast.
inline void tile_pack_scales_4x128_bf16(const uint16_t* scales_4x128_bf16,
                                        uint16_t* tiled_1kib_bf16) {
  std::memset(tiled_1kib_bf16, 0, 1024u);
  // Rows 0..1 map to sublane 0 (indices 0..255); rows 2..3 map to sublane 1 (256..511).
  for (uint32_t r = 0; r < 4u; ++r) {
    const uint32_t row_pair = r >> 1;
    const uint32_t parity = r & 1u;
    for (uint32_t c = 0; c < 128u; ++c) {
      tiled_1kib_bf16[row_pair * 256u + c * 2u + parity] =
          scales_4x128_bf16[r * 128u + c];
    }
  }
}

// Build constant 40 KiB HBM buffer required by quantized dequant IMEM kernels:
//   [0x0000..0x8000): Sel[128,128] BF16 tiled (1.0 in cols 0..3 for rows 0..31/32..63/64..95/96..127)
//   [0x8000..0x9000): C_NEG2304 (1,024 copies of -2304.0f = 0xc5100000u)
//   [0x9000..0xa000): C_NEG136  (1,024 copies of -136.0f  = 0xc3080000u)
inline void init_quant_const_vregs_40kib(uint8_t* out_40kib) {
  std::memset(out_40kib, 0, 40960u);
  std::vector<uint16_t> sel_rm(128u * 128u, 0u);
  for (uint32_t r = 0; r < 128u; ++r) {
    sel_rm[r * 128u + (r >> 5)] = 0x3f80u;  // 1.0f in BF16
  }
  tile_pack_128x128_bf16(sel_rm.data(), reinterpret_cast<uint16_t*>(out_40kib));

  uint32_t* neg2304 = reinterpret_cast<uint32_t*>(out_40kib + 32768u);
  uint32_t* neg136 = reinterpret_cast<uint32_t*>(out_40kib + 36864u);
  for (uint32_t i = 0; i < 1024u; ++i) {
    neg2304[i] = 0xc5100000u;  // -2304.0f
    neg136[i]  = 0xc3080000u;  // -136.0f
  }
}

namespace detail {

struct AsmBuilder {
  struct PendingBr {
    size_t   pc;
    uint32_t label;
    uint32_t pred5;
  };
  std::vector<VliwBundle> code;
  std::vector<size_t>     labels;
  std::vector<PendingBr>  branches;

  uint32_t new_label() {
    labels.push_back(static_cast<size_t>(-1));
    return static_cast<uint32_t>(labels.size() - 1u);
  }
  void bind(uint32_t lbl) { labels[lbl] = code.size(); }
  void push(const VliwBundle& b) { code.push_back(b); }
  void push_nops(size_t count) {
    for (size_t i = 0; i < count; ++i) code.push_back(VliwBundle::nop());
  }
  void emit_br_rel(uint32_t lbl, uint32_t pred5 = VliwBundle::kS0PredAlways) {
    const size_t pc = code.size();
    code.push_back(VliwBundle::nop());
    branches.push_back({pc, lbl, pred5});
    push_nops(4u);  // 4 architectural branch delay slots
  }
  void emit_s0_alu(uint32_t op6, uint32_t sd, uint32_t sx5, uint32_t sy6) {
    VliwBundle b = VliwBundle::nop();
    b.set_s0_alu(op6, sd, sx5, sy6);
    push(b);
  }
  void emit_s0_alu_imm(uint32_t op6, uint32_t sd, uint32_t sx5, uint32_t imm18) {
    VliwBundle b = VliwBundle::nop();
    b.set_s0_alu_imm(op6, sd, sx5, imm18);
    push(b);
  }
  void emit_s0_simm(uint32_t sd, uint32_t imm18) {
    VliwBundle b = VliwBundle::nop();
    b.set_s0_simm(sd, imm18);
    push(b);
  }
  void finalize_512b_aligned() {
    for (const PendingBr& br : branches) {
      const size_t target = labels[br.label];
      const int32_t rel = static_cast<int32_t>(target) - static_cast<int32_t>(br.pc);
      code[br.pc] = VliwBundle::make_br_rel(rel, br.pred5);
    }
    while ((code.size() % 8u) != 0u) {
      code.push_back(VliwBundle::nop());
    }
  }
};

inline void emit_resident_boot_stages_1_to_3(AsmBuilder& a) {
  // Stage 1 (PC 0..16): Read SMEM[+0x1e8] (exec_seq) -> sync[0x1f9].
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3b00060fu); b.set_s0(0x0f000000u, 0x55700dc4u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x2f000000u, 0x55701100u); a.push(b); }
  a.push_nops(1u);
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a00000fu); b.set_s1_imm(0xfffe8007u, 0xec000000u); b.set_s0(0x8e04100cu, 0x55714184u); a.push(b); }
  a.push_nops(1u);
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3b0e060fu); b.set_s1_imm(0x3fffc1ffu, 0xfc007dc0u); b.set_s0(0xcf000000u, 0x55700db0u); a.push(b); }
  a.push_nops(1u);
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xef000000u, 0x5571c60cu); a.push(b); }
  a.push_nops(5u);
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x6f000000u, 0x5571ea1du); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0a200fu); b.w[10] = 0x00000160u; b.set_s1_imm(kSyncFlagExecDone << 14); a.push(b); }
  a.push_nops(2u);

  // Stage 2 (PC 17..20): Wait & consume doorbell sync[0x1fb] >= 16, -= 16.
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x8f000000u, 0x55701101u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a82a30fu); b.w[10] = 0x00000080u; b.set_s1_imm(0x00040000u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a02a10fu); b.w[10] = 0x00000080u; b.set_s1_imm(0xfffc0000u); b.set_s0(0x0f000003u, 0x55780000u); a.push(b); }
  a.push_nops(1u);

  // Stage 3 (PC 21..27): Fetch 512 B OtV5eLaunchDesc from HBM 0x3fff00000 -> SMEM + wait/sub sync[0x1ff] + fence.
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(0x3e000007u, 0xfcffe000u); b.set_s0(0xae01b13cu, 0x55700db1u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0xcf000000u, 0x55714d31u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x0f000000u, 0x5572eed2u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(kSyncFlagLdescDma << 14); b.set_s0(0x0211f000u, 0x5570d0e4u); a.push(b); }
  a.push(VliwBundle::make_sync_wait_imm(kSyncFlagLdescDma, 512u));
  a.push(VliwBundle::make_sync_sub_imm(kSyncFlagLdescDma, 512u));
  a.push(VliwBundle::make_fence());
}

// Emit 8-chunk MXU weight push (v7 down to v0 from vmem_rhs), 4-strip activation
// push (v8..v15 from vmem_lhs), 8 MRF pops, and FP32 accumulation into v48..v63.
inline void emit_mxu_gemm_accum_v48_v63(AsmBuilder& a, uint32_t vmem_lhs_512b,
                                        uint32_t vmem_rhs_512b) {
  for (uint32_t t = 0; t < 8u; ++t) {
    a.push(VliwBundle::make_vld(t, vmem_rhs_512b + t * 8u));
  }
  for (uint32_t t = 0; t < 8u; ++t) {
    a.push(VliwBundle::make_vld(8u + t, vmem_lhs_512b + t * 8u));
  }
  for (int t = 7; t >= 0; --t) {
    for (int bank = 0; bank < 2; ++bank) {
      VliwBundle b = VliwBundle::nop();
      b.set_mxu_push_weight_vreg(bank, bank, static_cast<uint32_t>(t));
      a.push(b);
    }
  }
  static constexpr uint32_t kActSlots[4] = {4u, 5u, 2u, 3u};
  for (uint32_t i = 0; i < 4u; ++i) {
    VliwBundle b = VliwBundle::nop();
    b.set_mxu_push_act_vregs(kActSlots[i], 8u + 2u * i, 9u + 2u * i);
    a.push(b);
  }

  // Drain 8 MRF pops matching v5e systolic stagger while accumulating v48..v63.
  a.push(VliwBundle::make_mxu_pop(0, 16u, 17u));
  a.push(VliwBundle::make_mxu_pop(1, 18u, 19u));
  a.push(VliwBundle::make_mxu_pop(0, 20u, 21u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 48u, 48u, 16u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 49u, 49u, 17u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 50u, 50u, 18u));

  { VliwBundle b = VliwBundle::make_mxu_pop(1, 24u, 25u); b.set_v1_alu(VliwBundle::kV1OpAddF32, 51u, 51u, 19u); a.push(b); }
  { VliwBundle b = VliwBundle::make_mxu_pop(0, 28u, 29u); b.set_v1_alu(VliwBundle::kV1OpAddF32, 52u, 52u, 20u); a.push(b); }
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 53u, 53u, 21u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 54u, 54u, 24u));

  { VliwBundle b = VliwBundle::make_mxu_pop(1, 30u, 31u); b.set_v1_alu(VliwBundle::kV1OpAddF32, 55u, 55u, 25u); a.push(b); }
  { VliwBundle b = VliwBundle::make_mxu_pop(0, 32u, 33u); b.set_v1_alu(VliwBundle::kV1OpAddF32, 56u, 56u, 28u); a.push(b); }
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 57u, 57u, 29u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 58u, 58u, 30u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 59u, 59u, 31u));

  { VliwBundle b = VliwBundle::make_mxu_pop(1, 36u, 37u); b.set_v1_alu(VliwBundle::kV1OpAddF32, 60u, 60u, 32u); a.push(b); }
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 61u, 61u, 33u));
  a.push_nops(1u);
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 62u, 62u, 36u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 63u, 63u, 37u));
}

// Pack v48..v63 FP32 accumulators into 8 BF16 VREGs -> VMEM[0x100..0x138] -> DMA to HBM.
inline void emit_pack_and_store_tile_out(AsmBuilder& a, uint32_t s_out_hbm_dma,
                                         uint32_t s_vmem_out_dma, uint32_t flag_out,
                                         bool apply_per_channel_scale_v40) {
  if (apply_per_channel_scale_v40) {
    for (uint32_t i = 0; i < 16u; ++i) {
      a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpMulF32, 48u + i, 48u + i, 40u));
    }
    a.push_nops(2u);
  }
  struct PackSpec { uint32_t hi, lo, dst, vmem_off; };
  static constexpr PackSpec kPacks[8] = {
      {52u, 48u, 32u, 0x100u}, {53u, 49u, 33u, 0x108u},
      {54u, 50u, 34u, 0x110u}, {55u, 51u, 35u, 0x118u},
      {60u, 56u, 36u, 0x120u}, {61u, 57u, 37u, 0x128u},
      {62u, 58u, 38u, 0x130u}, {63u, 59u, 39u, 0x138u},
  };
  for (uint32_t i = 0; i < 8u; ++i) {
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, kPacks[i].dst,
                                   kPacks[i].hi, kPacks[i].lo));
  }
  a.push_nops(2u);
  for (uint32_t i = 0; i < 8u; ++i) {
    a.push(VliwBundle::make_vst(kPacks[i].vmem_off, kPacks[i].dst));
  }
  a.push(VliwBundle::make_fence());
  a.push(VliwBundle::make_dma_vmem_to_hbm_imm(s_out_hbm_dma, s_vmem_out_dma, flag_out, 1024u));
  a.push(VliwBundle::make_sync_wait_imm(flag_out, 32768u));
  a.push(VliwBundle::make_sync_sub_imm(flag_out, 32768u));
  a.push(VliwBundle::make_sync_clear(flag_out));
}

// Broadcast 4 K-block BF16 scales from VMEM[vmem_scale_512b] (rows 0..3; rows 4..15 zero)
// into v40..v43 (FP32 across all 8 sublanes) using selector matrix at VMEM[0x180..0x1b8].
inline void emit_mxu_broadcast_scales_v40_v43(AsmBuilder& a, uint32_t vmem_scale_512b,
                                              uint32_t v_zero) {
  a.push(VliwBundle::make_vld(0u, vmem_scale_512b));
  for (uint32_t t = 0; t < 8u; ++t) {
    a.push(VliwBundle::make_vld(8u + t, 0x180u + t * 8u));
  }
  for (int t = 7; t >= 1; --t) {
    for (int bank = 0; bank < 2; ++bank) {
      VliwBundle b = VliwBundle::nop();
      b.set_mxu_push_weight_vreg(bank, bank, v_zero);
      a.push(b);
    }
  }
  for (int bank = 0; bank < 2; ++bank) {
    VliwBundle b = VliwBundle::nop();
    b.set_mxu_push_weight_vreg(bank, bank, 0u);
    a.push(b);
  }
  static constexpr uint32_t kActSlots[4] = {4u, 5u, 2u, 3u};
  for (uint32_t i = 0; i < 4u; ++i) {
    VliwBundle b = VliwBundle::nop();
    b.set_mxu_push_act_vregs(kActSlots[i], 8u + 2u * i, 9u + 2u * i);
    a.push(b);
  }
  a.push(VliwBundle::make_mxu_pop(0, 16u, 17u));
  a.push(VliwBundle::make_mxu_pop(1, 18u, 19u));
  a.push(VliwBundle::make_mxu_pop(0, 20u, 21u));
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 40u, v_zero, 16u));  // rows 0..15 -> kb 0
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 41u, v_zero, 18u));  // rows 32..47 -> kb 1
  a.push_nops(1u);
  a.push(VliwBundle::make_mxu_pop(1, 24u, 25u));
  a.push(VliwBundle::make_mxu_pop(0, 28u, 29u));
  a.push(VliwBundle::make_mxu_pop(1, 30u, 31u));
  a.push(VliwBundle::make_mxu_pop(0, 32u, 33u));
  a.push_nops(2u);
  a.push(VliwBundle::make_mxu_pop(1, 36u, 37u));
  a.push_nops(2u);
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 42u, v_zero, 28u));  // rows 64..79 -> kb 2
  a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 43u, v_zero, 30u));  // rows 96..111 -> kb 3
  a.push_nops(2u);
}

// Dequantize 4 INT8 VREGs from VMEM[vmem_q_512b + kb*8] -> 8 BF16 VREGs at
// VMEM[0x040..0x078], optionally scaling each block kb by v(40+kb).
inline void emit_dequant_q8_to_vmem_040(AsmBuilder& a, uint32_t vmem_q_512b,
                                        uint32_t v_neg2304, bool scale_per_block32) {
  for (uint32_t kb = 0; kb < 4u; ++kb) {
    a.push(VliwBundle::make_vld(0u, vmem_q_512b + kb * 8u));
    a.push_nops(4u);
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 1u, 0u, 24));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 2u, 0u, 16));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 3u, 0u, 8));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShraS32Imm, 4u, 0u, 24));
    a.push_nops(3u);
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShraS32Imm, 1u, 1u, 24));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShraS32Imm, 2u, 2u, 24));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShraS32Imm, 3u, 3u, 24));
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 5u + lane, 1u + lane, 0xf));
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShraS32Imm, 9u + lane, 1u + lane, 4));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAddS32Imm, 5u + lane, 5u + lane, 0x4300));
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAddS32Imm, 9u + lane, 9u + lane, 0x4508));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 5u + lane, 5u + lane, 16));
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 9u + lane, 9u + lane, 16));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 1u + lane, 9u + lane, 5u + lane));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 1u + lane, 1u + lane, v_neg2304));
    }
    a.push_nops(3u);
    if (scale_per_block32) {
      for (uint32_t lane = 0; lane < 4u; ++lane) {
        a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpMulF32, 1u + lane, 1u + lane, 40u + kb));
      }
      a.push_nops(3u);
    }
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 5u, 2u, 1u));
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 6u, 4u, 3u));
    a.push_nops(3u);
    a.push(VliwBundle::make_vst(0x040u + kb * 16u, 5u));
    a.push(VliwBundle::make_vst(0x048u + kb * 16u, 6u));
  }
}

// Dequantize 2 Q4 VREGs from VMEM[vmem_q_512b + c4*8] -> 8 BF16 VREGs at
// VMEM[0x040..0x078], optionally scaling each 32-row block by v(40+kb).
inline void emit_dequant_q4_to_vmem_040(AsmBuilder& a, uint32_t vmem_q_512b,
                                        uint32_t v_neg136, bool scale_per_block32) {
  for (uint32_t c4 = 0; c4 < 2u; ++c4) {
    a.push(VliwBundle::make_vld(0u, vmem_q_512b + c4 * 8u));
    a.push_nops(4u);
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 5u, 0u, 0xf));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 2u, 0u, 8));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 3u, 0u, 16));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 4u, 0u, 24));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 9u, 0u, 4));
    a.push_nops(3u);
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 6u, 2u, 0xf));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 7u, 3u, 0xf));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 8u, 4u, 0xf));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 9u, 9u, 0xf));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 10u, 2u, 4));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 11u, 3u, 4));
    a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShrlU32Imm, 12u, 4u, 4));
    a.push_nops(3u);
    for (uint32_t lane = 1; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAndU32Imm, 9u + lane, 9u + lane, 0xf));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAddS32Imm, 5u + lane, 5u + lane, 0x4300));
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpAddS32Imm, 9u + lane, 9u + lane, 0x4300));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 5u + lane, 5u + lane, 16));
      a.push(VliwBundle::make_v1_alu_imm(VliwBundle::kV1OpShllU32Imm, 9u + lane, 9u + lane, 16));
    }
    a.push_nops(3u);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 5u + lane, 5u + lane, v_neg136));
      a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpAddF32, 9u + lane, 9u + lane, v_neg136));
    }
    a.push_nops(3u);
    const uint32_t kb_lo = c4 * 2u;
    const uint32_t kb_hi = c4 * 2u + 1u;
    if (scale_per_block32) {
      for (uint32_t lane = 0; lane < 4u; ++lane) {
        a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpMulF32, 5u + lane, 5u + lane, 40u + kb_lo));
        a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpMulF32, 9u + lane, 9u + lane, 40u + kb_hi));
      }
      a.push_nops(3u);
    }
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 13u, 6u, 5u));
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 14u, 8u, 7u));
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 15u, 10u, 9u));
    a.push(VliwBundle::make_v1_alu(VliwBundle::kV1OpPackF32Bf16, 16u, 12u, 11u));
    a.push_nops(3u);
    a.push(VliwBundle::make_vst(0x040u + kb_lo * 16u, 13u));
    a.push(VliwBundle::make_vst(0x048u + kb_lo * 16u, 14u));
    a.push(VliwBundle::make_vst(0x040u + kb_hi * 16u, 15u));
    a.push(VliwBundle::make_vst(0x048u + kb_hi * 16u, 16u));
  }
}

}  // namespace detail

// Emit FreeTile-aligned bare-metal multi-tile BF16 resident IMEM kernel
// supporting any (M, K, N) multiples of 128 via triple M/N/K loops, ping-pong
// HBM<->VMEM double-buffering (sync flags 51/52), and FP32 K-accumulation in v48..v63.
inline std::vector<VliwBundle> emit_v5e_resident_freetile_bf16_imem() {
  using detail::AsmBuilder;
  AsmBuilder a;
  detail::emit_resident_boot_stages_1_to_3(a);

  const uint32_t lbl_has_stride  = a.new_label();
  const uint32_t lbl_single_tile = a.new_label();
  const uint32_t lbl_multitile   = a.new_label();
  const uint32_t lbl_m_loop      = a.new_label();
  const uint32_t lbl_n_loop      = a.new_label();
  const uint32_t lbl_k_ping      = a.new_label();
  const uint32_t lbl_skip_pf1    = a.new_label();
  const uint32_t lbl_k_pong      = a.new_label();
  const uint32_t lbl_skip_pf0    = a.new_label();
  const uint32_t lbl_tile_epi    = a.new_label();
  const uint32_t lbl_done        = a.new_label();

  // Load launch descriptor parameters.
  a.push(VliwBundle::make_smem_load_ldesc(2u,  0x190u));  // s2  = out_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(18u, 0x174u));  // s18 = lhs_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(1u,  0x170u));  // s1  = rhs_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(19u, 0x178u));  // s19 = m_tiles
  a.push(VliwBundle::make_smem_load_ldesc(20u, 0x180u));  // s20 = k_tiles
  a.push(VliwBundle::make_smem_load_ldesc(21u, 0x184u));  // s21 = n_tiles (chunk width)
  a.push(VliwBundle::make_smem_load_ldesc(3u,  0x198u));  // s3  = n_stride_tiles (0 -> n_tiles)
  a.push_nops(6u);

  a.emit_s0_alu_imm(VliwBundle::kS0OpEq, 0u, 3u, 0u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_has_stride, 0x10u);  // @!p0
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 3u, 21u, VliwBundle::kSyConst0);
  a.bind(lbl_has_stride);

  // Fast path: if (m_tiles == 1 && k_tiles == 1 && n_tiles == 1 && n_stride <= 1), branch to single-tile body.
  a.emit_s0_alu_imm(VliwBundle::kS0OpLt, 0u, 3u, 2u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_multitile, 0x10u);  // @!p0
  a.emit_s0_alu_imm(VliwBundle::kS0OpEq, 0u, 19u, 1u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_multitile, 0x10u);  // @!p0
  a.emit_s0_alu_imm(VliwBundle::kS0OpEq, 0u, 20u, 1u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_multitile, 0x10u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpEq, 0u, 21u, 1u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_single_tile, 0x00u);  // @p0

  a.bind(lbl_multitile);
  // Precompute 32B DMA addresses & strides: s1=rhs_base, s2=out_base, s18=lhs_base.
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 1u, 1u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 2u, 2u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 18u, 18u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 25u, 3u, 10u);   // s25 = n_stride_tiles * 0x400 (rhs K & out M stride)
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 26u, 20u, 10u);  // s26 = k_tiles * 0x400 (lhs M-stride)
  a.emit_s0_simm(14u, 0x0000u);  // s14 = VMEM 0x000 DMA (LHS ping)
  a.emit_s0_simm(15u, 0x0400u);  // s15 = VMEM 0x040 DMA (RHS ping)
  a.emit_s0_simm(27u, 0x0800u);  // s27 = VMEM 0x080 DMA (LHS pong)
  a.emit_s0_simm(28u, 0x0c00u);  // s28 = VMEM 0x0c0 DMA (RHS pong)
  a.emit_s0_simm(29u, 0x1000u);  // s29 = VMEM 0x100 DMA (OUT)
  a.emit_s0_simm(4u, 0u);        // s4 = m_idx = 0

  // Outer M loop.
  a.bind(lbl_m_loop);
  a.emit_s0_alu(VliwBundle::kS0OpMul, 7u, 4u, 26u);
  a.push_nops(2u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 7u, 18u, 7u);   // s7 = lhs_m_base
  a.emit_s0_alu(VliwBundle::kS0OpMul, 8u, 4u, 25u);
  a.push_nops(2u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 8u, 2u, 8u);    // s8 = out_m_base
  a.emit_s0_simm(5u, 0u);                             // s5 = n_idx = 0

  // Middle N loop.
  a.bind(lbl_n_loop);
  for (uint32_t i = 0; i < 16u; ++i) {
    a.push(VliwBundle::make_v1_zero(48u + i));
  }
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 9u, 7u, VliwBundle::kSyConst0);   // s9 = lhs_cur = lhs_m_base
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 10u, 5u, 10u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 11u, 5u, 10u);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 10u, 1u, 10u);  // s10 = rhs_cur = rhs_base + n_idx*0x400
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 30u, 8u, 11u);  // s30 = out_mn_dma
  a.push_nops(1u);

  // Kick k = 0 into PING (VMEM 0x000, 0x040) on sync flag 51.
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(14u, 9u, 51u, 1024u));
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(15u, 10u, 51u, 1024u));
  a.emit_s0_simm(6u, 0u);  // s6 = k_idx = 0

  // Inner K loop — PING step.
  a.bind(lbl_k_ping);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 17u, 6u, VliwBundle::kSyConst1);  // s17 = k_next
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 17u, 20u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_skip_pf1, 0x10u);  // @!p0
  a.emit_s0_alu_imm(VliwBundle::kS0OpAdd, 11u, 9u, 0x400u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 12u, 10u, 25u);
  a.push_nops(1u);
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(27u, 11u, 52u, 1024u));
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(28u, 12u, 52u, 1024u));
  a.bind(lbl_skip_pf1);

  a.push(VliwBundle::make_sync_wait_imm(51u, 65536u));
  a.push(VliwBundle::make_sync_sub_imm(51u, 65536u));
  a.push(VliwBundle::make_sync_clear(51u));
  detail::emit_mxu_gemm_accum_v48_v63(a, 0x000u, 0x040u);

  a.emit_s0_alu(VliwBundle::kS0OpAdd, 6u, 17u, VliwBundle::kSyConst0);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 6u, 20u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_tile_epi, 0x10u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 9u, 11u, VliwBundle::kSyConst0);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 10u, 12u, VliwBundle::kSyConst0);

  // Inner K loop — PONG step.
  a.bind(lbl_k_pong);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 17u, 6u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 17u, 20u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_skip_pf0, 0x10u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpAdd, 11u, 9u, 0x400u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 12u, 10u, 25u);
  a.push_nops(1u);
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(14u, 11u, 51u, 1024u));
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(15u, 12u, 51u, 1024u));
  a.bind(lbl_skip_pf0);

  a.push(VliwBundle::make_sync_wait_imm(52u, 65536u));
  a.push(VliwBundle::make_sync_sub_imm(52u, 65536u));
  a.push(VliwBundle::make_sync_clear(52u));
  detail::emit_mxu_gemm_accum_v48_v63(a, 0x080u, 0x0c0u);

  a.emit_s0_alu(VliwBundle::kS0OpAdd, 6u, 17u, VliwBundle::kSyConst0);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 6u, 20u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_tile_epi, 0x10u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 9u, 11u, VliwBundle::kSyConst0);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 10u, 12u, VliwBundle::kSyConst0);
  a.emit_br_rel(lbl_k_ping);

  // Store finished (M_tile, N_tile) BF16 output & advance N / M.
  a.bind(lbl_tile_epi);
  detail::emit_pack_and_store_tile_out(a, 30u, 29u, 54u, /*apply_pc=*/false);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 5u, 5u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 5u, 21u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_n_loop, 0x00u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 4u, 4u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 4u, 19u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_m_loop, 0x00u);
  a.emit_br_rel(lbl_done);

  // Fast single-tile 128x128x128 BF16 path preserving proven PC 37..108 behavior.
  a.bind(lbl_single_tile);
  { VliwBundle b = VliwBundle::nop(); b.set_dma_sync(0x3a0a210fu); b.w[10] = 0x00000320u; b.set_s1_imm(kSyncFlagLhsDma << 14); b.set_s0(0x4e01bee8u, 0x55780000u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x6e01b874u, 0x5572f2d3u); a.push(b); }
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x8e5fba78u, 0x5572fad3u); a.push(b); }
  a.push_nops(5u);
  { VliwBundle b = VliwBundle::nop(); b.set_s1_imm(kSyncFlagLhsDma << 14); b.set_s0(0x0013d000u, 0x5570db72u); a.push(b); }
  a.push(VliwBundle::make_sync_wait_imm(kSyncFlagLhsDma, 32768u));
  a.push(VliwBundle::make_sync_sub_imm(kSyncFlagLhsDma, 32768u));
  a.push(VliwBundle::make_sync_clear(kSyncFlagLhsDma));
  { VliwBundle b = VliwBundle::nop(); b.set_s0(0x0f000000u, 0x55700dc0u); a.push(b); }
  a.push_nops(1u);
  const std::vector<VliwBundle> body = emit_v5e_matmul_128x128_body();
  a.code.insert(a.code.end(), body.begin(), body.end());

  a.bind(lbl_done);
  // Zero scalar state so Stage 1 (PC 0..16) observes clean registers after br_abs(0).
  for (uint32_t sr = 0u; sr < 32u; ++sr) {
    a.emit_s0_simm(sr, 0u);
  }
  a.push(VliwBundle::make_sync_clear(kSyncFlagLhsDma));
  a.push(VliwBundle::make_sync_clear(52u));
  a.push(VliwBundle::make_sync_clear(53u));
  a.push(VliwBundle::make_sync_clear(54u));
  a.push(VliwBundle::make_fence());
  a.push(VliwBundle::make_br_abs(0u));
  a.push_nops(4u);

  a.finalize_512b_aligned();
  return a.code;
}

// Emit FreeTile-aligned bare-metal quantized multi-tile IMEM kernel for either
// Q8/INT8 (is_q4=false) or Q4/INT4 (is_q4=true). Supports block-32 (Q8_0/Q4_0,
// quant_mode 1/2) and per-channel (INT8/INT4, quant_mode 3/4) via shared VPU
// exact IEEE-754 dequantization + MXU scale broadcast into canonical VMEM[0x040..0x078].
inline std::vector<VliwBundle> emit_v5e_resident_freetile_quant_imem(bool is_q4) {
  using detail::AsmBuilder;
  AsmBuilder a;
  detail::emit_resident_boot_stages_1_to_3(a);

  const uint32_t lbl_has_stride = a.new_label();
  const uint32_t lbl_m_loop     = a.new_label();
  const uint32_t lbl_n_loop     = a.new_label();
  const uint32_t lbl_k_loop     = a.new_label();
  const uint32_t lbl_is_pc      = a.new_label();
  const uint32_t lbl_sc_done    = a.new_label();
  const uint32_t lbl_done       = a.new_label();

  const uint32_t q_units32 = is_q4 ? 256u : 512u;   // 8 KiB (Q4) vs 16 KiB (Q8)
  const uint32_t q_bytes   = is_q4 ? 8192u : 16384u;
  const uint32_t q_shift   = is_q4 ? 8u : 9u;
  const uint32_t const_vreg_off = is_q4 ? 0x1d0u : 0x1c0u;  // -136 vs -2304

  a.push(VliwBundle::make_smem_load_ldesc(2u,  0x190u));  // s2  = out_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(18u, 0x174u));  // s18 = lhs_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(1u,  0x170u));  // s1  = rhs_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(19u, 0x178u));  // s19 = m_tiles
  a.push(VliwBundle::make_smem_load_ldesc(20u, 0x180u));  // s20 = k_tiles
  a.push(VliwBundle::make_smem_load_ldesc(21u, 0x184u));  // s21 = n_tiles (chunk width)
  a.push(VliwBundle::make_smem_load_ldesc(22u, 0x188u));  // s22 = quant_mode
  a.push(VliwBundle::make_smem_load_ldesc(23u, 0x18cu));  // s23 = scales_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(24u, 0x194u));  // s24 = const_hbm >> 9
  a.push(VliwBundle::make_smem_load_ldesc(3u,  0x198u));  // s3  = n_stride_tiles
  a.push_nops(6u);

  a.emit_s0_alu_imm(VliwBundle::kS0OpEq, 0u, 3u, 0u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_has_stride, 0x10u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 3u, 21u, VliwBundle::kSyConst0);
  a.bind(lbl_has_stride);

  // Convert bases to 32B DMA units.
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 1u,  1u,  4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 2u,  2u,  4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 18u, 18u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 23u, 23u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 24u, 24u, 4u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 25u, 3u,  q_shift);  // rhs_k_stride
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 26u, 20u, 10u);      // lhs_m_stride
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 27u, 3u,  10u);      // out_m_stride
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 28u, 3u,  5u);       // scale_k_stride (n_stride_tiles * 1KiB)

  // Stage constants (Sel 32KiB + neg-const 4KiB) once into VMEM[0x180..], zero VMEM[0x140..0x178].
  a.emit_s0_simm(13u, 0x1800u);  // Sel[128,128] VMEM DMA
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(13u, 24u, 51u, 1024u));
  a.emit_s0_alu_imm(VliwBundle::kS0OpAdd, 11u, 24u, is_q4 ? 0x480u : 0x400u);  // +36KiB / +32KiB
  a.emit_s0_simm(12u, is_q4 ? 0x1d00u : 0x1c00u);
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(12u, 11u, 51u, 128u));
  a.push(VliwBundle::make_v1_zero(45u));
  a.push_nops(4u);
  for (uint32_t sv = 0u; sv < 8u; ++sv) {
    a.push(VliwBundle::make_vst(0x140u + sv * 8u, 45u));
  }
  a.push(VliwBundle::make_fence());
  a.push(VliwBundle::make_sync_wait_imm(51u, 36864u));
  a.push(VliwBundle::make_sync_sub_imm(51u, 36864u));
  a.push(VliwBundle::make_sync_clear(51u));
  a.push(VliwBundle::make_vld(44u, const_vreg_off));

  a.emit_s0_simm(14u, 0x0000u);  // LHS VMEM DMA
  a.emit_s0_simm(15u, 0x0c00u);  // Q   VMEM DMA (0x0c0)
  a.emit_s0_simm(16u, 0x1400u);  // SC  VMEM DMA (0x140)
  a.emit_s0_simm(29u, 0x1000u);  // OUT VMEM DMA (0x100)
  a.emit_s0_simm(4u, 0u);        // m_idx

  a.bind(lbl_m_loop);
  a.emit_s0_alu(VliwBundle::kS0OpMul, 7u, 4u, 26u);
  a.push_nops(2u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 7u, 18u, 7u);
  a.emit_s0_alu(VliwBundle::kS0OpMul, 8u, 4u, 27u);
  a.push_nops(2u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 8u, 2u, 8u);
  a.emit_s0_simm(5u, 0u);  // n_idx

  a.bind(lbl_n_loop);
  for (uint32_t i = 0; i < 16u; ++i) {
    a.push(VliwBundle::make_v1_zero(48u + i));
  }
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 9u, 7u, VliwBundle::kSyConst0);  // lhs_cur
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 10u, 5u, q_shift);
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 11u, 5u, 5u);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 10u, 1u, 10u);                   // rhs_cur
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 12u, 23u, 11u);                  // scale_cur
  a.emit_s0_alu_imm(VliwBundle::kS0OpShll, 11u, 5u, 10u);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 30u, 8u, 11u);                   // out_mn_dma
  a.emit_s0_simm(6u, 0u);                                              // k_idx
  a.push_nops(1u);

  a.bind(lbl_k_loop);
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(14u, 9u, 51u, 1024u));       // 32 KiB LHS
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(15u, 10u, 51u, q_units32));  // 8/16 KiB Q
  a.push(VliwBundle::make_dma_hbm_to_vmem_imm(16u, 12u, 51u, 32u));        // 1 KiB scales
  const uint32_t dma_bytes = 32768u + q_bytes + 1024u;
  a.push(VliwBundle::make_sync_wait_imm(51u, dma_bytes));
  a.push(VliwBundle::make_sync_sub_imm(51u, dma_bytes));
  a.push(VliwBundle::make_sync_clear(51u));

  // Broadcast 4 K-block BF16 scales from VMEM[0x140] -> v40..v43 (skip step for per-channel >=3 if desired,
  // or keep identical when s28 stride=0 and scales row0..3 all repeat per-channel scale).
  detail::emit_mxu_broadcast_scales_v40_v43(a, 0x140u, 45u);

  if (is_q4) {
    detail::emit_dequant_q4_to_vmem_040(a, 0x0c0u, 44u, /*scale_per_block32=*/true);
  } else {
    detail::emit_dequant_q8_to_vmem_040(a, 0x0c0u, 44u, /*scale_per_block32=*/true);
  }

  detail::emit_mxu_gemm_accum_v48_v63(a, 0x000u, 0x040u);

  a.emit_s0_alu_imm(VliwBundle::kS0OpAdd, 9u, 9u, 0x400u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 10u, 10u, 25u);
  a.emit_s0_alu_imm(VliwBundle::kS0OpGe, 0u, 22u, 3u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_is_pc, 0x00u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 12u, 12u, 28u);  // block-32 scale advance
  a.emit_br_rel(lbl_sc_done);
  a.bind(lbl_is_pc);
  a.push_nops(1u);  // per-channel keeps s12 fixed on n_tile
  a.bind(lbl_sc_done);

  a.emit_s0_alu(VliwBundle::kS0OpAdd, 6u, 6u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 6u, 20u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_k_loop, 0x00u);

  detail::emit_pack_and_store_tile_out(a, 30u, 29u, 54u, /*apply_pc=*/false);

  a.emit_s0_alu(VliwBundle::kS0OpAdd, 5u, 5u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 5u, 21u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_n_loop, 0x00u);
  a.emit_s0_alu(VliwBundle::kS0OpAdd, 4u, 4u, VliwBundle::kSyConst1);
  a.push_nops(1u);
  a.emit_s0_alu(VliwBundle::kS0OpLt, 0u, 4u, 19u);
  a.push_nops(1u);
  a.emit_br_rel(lbl_m_loop, 0x00u);

  a.bind(lbl_done);
  for (uint32_t sr = 0u; sr < 32u; ++sr) {
    a.emit_s0_simm(sr, 0u);
  }
  a.push(VliwBundle::make_sync_clear(kSyncFlagLhsDma));
  a.push(VliwBundle::make_sync_clear(52u));
  a.push(VliwBundle::make_sync_clear(53u));
  a.push(VliwBundle::make_sync_clear(54u));
  a.push(VliwBundle::make_fence());
  a.push(VliwBundle::make_br_abs(0u));
  a.push_nops(4u);

  a.finalize_512b_aligned();
  return a.code;
}

// Backend B FreeTile LoweringPlan entry point: emits the exact v5e-1 IMEM binary
// for the validated FreeTile lowering schedule.
inline std::vector<VliwBundle> emit_v5e_imem_for_freetile(
    const ot::freetile::LoweringPlan& plan) {
  switch (plan.desc.rhs_dtype) {
    case ot::freetile::DType::kInt8:
    case ot::freetile::DType::kQ8_0:
      return emit_v5e_resident_freetile_quant_imem(/*is_q4=*/false);
    case ot::freetile::DType::kQ4_0:
    case ot::freetile::DType::kQ4_K:
      return emit_v5e_resident_freetile_quant_imem(/*is_q4=*/true);
    case ot::freetile::DType::kBf16:
    default:
      return emit_v5e_resident_freetile_bf16_imem();
  }
}

}  // namespace ot

#endif  // OPENTPU_VLIW_H_
