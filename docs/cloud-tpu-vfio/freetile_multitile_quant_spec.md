# FreeTile-Aligned Multi-Tile BF16 & Bare-Metal Quantized GEMM on Cloud TPU `v5e-1` (`/dev/vfio/0`)

Clean-room specification and live silicon proof for extending `libopentpu` (`0x1ae0:0x0063`, zero `libtpu.so`, zero `libpython`) with:
1. A zero-dependency C11/C++17 **`FreeTile` Backend B** descriptor, VMEM double-buffer planner, and IMEM emitter (`ot_freetile_gemm_desc_t`, [`opentpu_freetile.hpp`](../../libopentpu/include/opentpu_freetile.hpp)).
2. Multi-tile `(M, K, N)` `BF16` GEMM (`ot_matmul_bf16_ex`) with triple M/N/K loops, ping-pong VMEM DMA double-buffering (`sync[51]` / `sync[52]`), and `FP32` MXU accumulation (`v48..v63`).
3. Bare-metal `Q8_0` / `Q4_0` (block-32) and per-channel `INT8` / `INT4` unpack + IEEE-754 VPU dequantization + MXU scale broadcast + matmul (`ot_matmul_quant`) directly in IMEM VLIW bundles.

---

## 1. FreeTile Descriptor & Lowering Seam

The C11 ABI in [`opentpu.h`](../../libopentpu/include/opentpu.h) exposes `ot_freetile_gemm_desc_t` and `ot_exec_stats_t`, mirrored in pure C++17 by [`opentpu_freetile.hpp`](../../libopentpu/include/opentpu_freetile.hpp):
- `ot::freetile::TileKernelDescriptor::from_c_desc(...)` captures `(m_dim, k_dim, n_dim)`, `(block_m=128, block_k=128, block_n=128)`, `vmem_double_buffer`, `k_loop_accum_f32`, `rhs_transposed`, `quant_mode`, `quant_group_size`, `scale_dtype`, and scale parameters.
- `ot::freetile::plan_v5e_tile_gemm(desc)` produces a deterministic `LoweringPlan` with exact VMEM buffer map (`0x000..0x1f0`), K-loop double-buffer footprint, HBM DMA traffic estimate (including 1 KiB scale tile loads per `(m_tile, n_tile, k_tile)` iteration and 36 KiB constant loads per launch chunk when quantized), and dispatch entry point (`ot::emit_v5e_imem_for_freetile(plan)`).
- `ot_matmul_quant` accepts either split weights + separate `rhs_scales` (element type explicitly controlled by `desc->scale_dtype`: `OT_DTYPE_BF16` or `OT_DTYPE_F32`) or inline interleaved `ot_block_q8_0_t` / `ot_block_q4_0_t` records (`rhs_scales == NULL` for `Q8_0`/`Q4_0` `BLOCK_32`). `OT_DTYPE_Q4_K` and `OT_QUANT_BLOCK_256` are reserved and return `OT_ERR_UNSUPPORTED`.

### Launch Descriptor (`OtV5eLaunchDesc` at HBM `0x3fff00000`)

Extended 512-byte launch descriptor fields read by the resident IMEM kernels:

| Byte Offset | Scalar Reg | Meaning |
| --- | --- | --- |
| `+0x170` | `s1` | `rhs_hbm >> 9` |
| `+0x174` | `s18` | `lhs_hbm >> 9` |
| `+0x178` | `s19` | `m_tiles` (`M / 128`) |
| `+0x180` | `s20` | `k_tiles` (`K / 128`) |
| `+0x184` | `s21` | `n_tiles` (`N / 128`, or chunk N width) |
| `+0x188` | `s22` | `quant_mode` (`0`=BF16, `1`=Q8_0 blk32, `2`=Q4_0 blk32, `3`=INT8 pc, `4`=INT4 pc) |
| `+0x18c` | `s23` | `scales_hbm >> 9` |
| `+0x190` | `s2` | `out_hbm >> 9` |
| `+0x194` | `s24` | `const_hbm >> 9` (40 KiB table: `Sel[128,128]` + IEEE-754 dequant bias VREGs) |
| `+0x198` | `s3` | `n_stride_tiles` (0 defaults to `s21`; enables bounded sub-chunking across `N`) |
| `+0x1e8` | — | `exec_seq` echoed into `sync[0x1f9]` by Stage 1 |

---

## 2. Clean-Room Hardware Invariants Established on Live `v5e-1`

Every non-obvious ISA/VFIO constraint below was falsified directly on live `v5e-1` silicon:

1. **256-Slot SQ/CQ Ring Wraparound (`ot_vfio.cpp`)**:
   - `BAR0+0x10018` (`SQ_TAIL`), `BAR0+0x1002c` (`CQ_HEAD`), `status_page_mem[0]` (`SQ_HEAD`), and `status_page_mem[1]` (`CQ_TAIL`) wrap **modulo 256** (`[0..255]`), while `sq_slot[0]` and `cq_slot[0]` retain the full 32-bit monotonic `sent_seq`. Writing raw `256` to `0x10018` stalls slot `255` (`SQ doorbell timeout ... seq=255`). Wrapping `(sent_seq + 1) % 256` for MMIO pointers allows unbounded consecutive submissions.
2. **Non-Blocking `OP_EXEC` Sync Poll (`ot_vfio.cpp`)**:
   - `kSqOpcodeExec` (`0x06`, `w8=0x41`, `w10=0x010101f9`) is a **one-shot sync-flag poll** (`cq[4] = (sync[0x1f9] >= 1)`, `cq[5] = sync[0x1f9]`) that does not clear `sync[0x1f9]`. When multi-tile/quantized kernels run longer than the host's `dma_h2d_sync` -> `submit_exec_sync` latency, `submit_exec_sync` must poll with bounded backoff until `cq[4] == 1 && cq[5] == expected_exec_seq`.
3. **Scalar Register Hygiene Across `br_abs(0)`**:
   - Stage 1 (`PC 0..16`) reads scalar register state (`s10`) before signaling completion. All `s0..s31` must be cleared in `lbl_done` prior to `make_fence()` + `make_br_abs(0)` so consecutive resident executions stay clean.
4. **Vector / MXU Encodings (`ot_vliw.h`)**:
   - `kSyConst4` is not constant `4` on `v5e-1`; immediate shift `kS0OpShll(..., 4u)` must be used.
   - `make_v1_zero(vd)` uses `vand.u32 vd, vd, $0` (`kV1OpAndU32Imm`).
   - Linear `kPacks[0..7]` store to `0x100u + i * 8u` matches `{1,0:T(2,128)(2,1)}` BF16 tile stores.
   - `kV1OpPackF32Bf16(dst, vx, vy)` packs adjacent sublane pairs `(2s, 2s+1)` of `vy` (`sl0..3`) then `vx` (`sl4..7`).
   - Uniform 8-sublane bitfield shifts/masks (`0x4807`, `0x5007`, `0x5807`, `0x3007`) unpack `Q8_0` (`int8`) and `Q4_0` (`nibble - 8`) cleanly into IEEE-754 `0x4300` magic-bias float conversion and `v40..v43` MXU-broadcast `K`-block scales (populated via `v16/v18/v28/v30` -> `VMEM[0x140..0x178]`).

---

## 3. Live `v5e-1` Silicon Proof vs `libtpu.so` XLA Baseline

Measured on live Google Colab `v5e-1` (`0x1ae0:0x0063`, `/dev/vfio/0`):
- `libopentpu` command: `./libopentpu/build/host/opentpu_matmul_proof --runs 3 --multitile --quant`
- Runtime maps: **`MAPS_LIBTPU=0`**, **`MAPS_LIBPYTHON=0`** (`NEEDED`: `libstdc++.so.6`, `libc.so.6`, `ld-linux-x86-64.so.2` only)
- XLA baseline: `jax` (`libtpu.so`) `jnp.matmul` steady-state median latency via `block_until_ready()` in an isolated child process.

> **Bandwidth definitions**:
> - **Unique Footprint GB/s** = `(bytes(A) + bytes(B_packed) + bytes(scales) + bytes(C)) / time_us`, comparable directly across XLA and `libopentpu`.
> - **DMA Traffic GB/s** = `plan.hbm_bytes_moved / hw_us` (accounts for HBM->VMEM re-reads in the triple-loop tile schedule).

| Shape & Format | `libopentpu` Mismatches (vs FP32 Ref) | `libopentpu` `hw_us` | XLA `median_us` (`libtpu.so`) | `libopentpu` Footprint GB/s | `libopentpu` DMA GB/s | XLA Footprint GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| `BF16` `128x512x512` | `0 / 65,536` (`max_diff=0.0`) | `136 us` | `124.8 us` | `5.78` | `8.67` | `6.30` |
| `BF16` `256x512x256` | `0 / 65,536` (`max_diff=0.0`) | `134 us` | `132.8 us` | `4.89` | `8.80` | `4.93` |
| `BF16` `128x4096x4096` | `0 / 524,288` (`max_diff=0.0`) | `435 us` | `142.2 us` | `81.96` | `156.68` | `250.71` |
| `Q8_0` blk-32 `128x512x512` | `0 / 65,536` (`max_diff=0.0`) | `135 us` | `141.8 us` | `4.00` | `6.80` | `3.81` |
| `Q4_0` blk-32 `128x512x512` | `0 / 65,536` (`max_diff=0.0`) | `134 us` | `227.7 us` | `3.06` | `5.87` | `1.80` |
| `Q8_0` blk-32 `128x4096x4096` | `0 / 524,288` (`max_diff=0.0`) | `1,153 us` | `250.1 us` | `17.28` | `44.56` | `79.67` |
| `Q4_0` blk-32 `128x4096x4096` | `0 / 524,288` (`max_diff=0.0`) | `1,026 us` | `453.3 us` | `11.24` | `41.90` | `25.45` |

On small/medium `512x512` quantized matmuls, `libopentpu`'s resident bare-metal IMEM path already beats `libtpu.so` XLA latency (`134 us` vs `227.7 us`, **`1.70x` faster** on `Q4_0`). At `4096x4096`, `libopentpu` reaches **`156.7–166.6 GB/s` DMA (`82–87 GB/s` footprint)** in BF16 and **`41.9–50.3 GB/s` DMA** in quantized modes with zero `libtpu.so`; adding ping-pong DMA overlap inside `emit_v5e_resident_freetile_quant_imem` is the natural next throughput step.
