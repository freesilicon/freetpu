# Cloud TPU `v5e-1` Clean-Room Direct-in-IMEM Boot/Poller & Matmul Architecture

Clean-room specification of the self-contained 120-bundle (`7,680 B`) resident IMEM boot/poller and `128x128` BF16 MXU matmul loop implemented in `libopentpu/src/ot_vliw.h` (`ot::emit_v5e_resident_matmul_imem()`), `libopentpu/src/ot_vfio.cpp`, and `libopentpu/src/ot_device.cpp`.

Verified directly on live Cloud TPU `v5e-1` (`0000:00:05.0`, PCI `0x1ae0:0x0063`) silicon over `/dev/vfio/0` with zero proprietary blobs, zero `libtpu.so`, and zero `libpython` (`MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`).

---

## 1. Architecture Overview

Rather than relying on an external HBM program reload (`prog_hbm`) or a multi-slot descriptor ring, `libopentpu` uploads a **120-bundle (`7,680 B` = `15 x 512 B`) resident VLIW loop** directly from host staging memory to IMEM `PC=0` once per device open/reset and pins the 512-byte `OtV5eLaunchDesc` to fixed slot 0 (`0x3fff00000`).

### 1.1 One-Time Bring-Up (`VfioTransport::ensure_imem_booted`)

Executed lazily on the first live `ot_matmul_bf16` call (idempotent across calls; reset on `teardown_ring_queues` / `close_device`):

1. **Minimal VFIO Ring Setup (`init_ring_queues`)**: `VFIO_DEVICE_RESET`, enable `PCI_COMMAND_MASTER`, map SQ (`0x0`, 32 KiB), CQ (`0x8000`, 8 KiB), Status Page (`0xa000`, 4 KiB), and 4 MiB Host DMA Staging (`0x100000`) via `VFIO_IOMMU_MAP_DMA`, and reset `BAR0 + 0x10000` ring pointers (`0x10018 = 0`, `0x1002c = 0`, `0x10044 = 0`). Zero `BAR0 + 0x0a...` CSR writes are required.
2. **Direct Host -> IMEM DMA (`dma_host_to_imem`)**:
   - Emit 120 `VliwBundle`s (`7,680 B`) into `host_stage_mem`.
   - Submit `op=0x04` SQ DMA (`dst_addr = 0`, `len = 7680`, `is_h2d = true`, `dst_flags_hi = 0x10000000`, `w28 = 0x130`, `w29 = 0x04000000`).
   - **Alignment Invariant**: IMEM DMA length **must** be a positive multiple of `512 B` (`8` bundles); unaligned transfers (e.g. `100` bundles = `6,400 B`) time out on silicon, while `512 B`-aligned lengths (`104/112/120/128` bundles) succeed.
3. **Start Scalar Core (`submit_core_ctrl`)**:
   - Submit `op=0x03` (`d[1] = 0x03`, `d[8] = 0x121`). On cold start, the core executes Stage 1 (`SMEM[+0x1e8] = 0 -> sync[0x1f9]`) and parks at Stage 2 (`PC 18`) waiting for host doorbell `sync[0x1fb]`.

### 1.2 Per-Call Matmul Dispatch (`ot_matmul_bf16` & `GetPjrtApi`)

1. **HBM `{1,0:T(2,128)}` Tensor Layout Management**:
   - The v5e MXU kernel operates on `{1,0:T(2,128)}` pair-major BF16 tiles (`tiled_u16_idx(r, c) = (r >> 1) * 256 + c * 2 + (r & 1)`).
   - Every `ot_buffer_t` tracks its current on-device layout (`BufferLayout::kRaw` vs `BufferLayout::kTiled2x128Bf16`).
   - `ot_pjrt.cpp` (`PJRT_Client_BufferFromHostBuffer`) uploads `128x128` BF16 tensors directly in `{1,0:T(2,128)}` layout (`ot::buffer_h2d_tiled_128x128_bf16`), avoiding any repack DMA during `PJRT_LoadedExecutable_Execute`.
   - Raw C API buffers uploaded via `ot_buffer_h2d` (`kRaw`) are repacked into persistent 32 KiB HBM scratch slots (`scratch_lhs_hbm` / `scratch_rhs_hbm`) without mutating `const` inputs. In-place aliasing (`out == lhs` or `out == rhs`) also isolates inputs into scratch, while self-multiply (`lhs == rhs`) shares a single tiled HBM address.
   - Outputs are written directly in `{1,0:T(2,128)}` layout and marked `kTiled2x128Bf16`, allowing chained device-resident matmuls `(A @ B) @ D` with zero intermediate repack and transparent row-major unpacking in `ot_buffer_d2h` / `PJRT_Buffer_ToHostBuffer`.
2. **Launch Descriptor (`OtV5eLaunchDesc`, `512 B` at `0x3fff00000`)**:
   - Monotonically increment `exec_seq` (`1, 2, 3, ...`, wrapping `0xffffffff -> 1`).
   - Four `uint32_t` words are causally consumed by the resident loop (`prog_hbm = 0`):
     - `+0x170`: `rhs_hbm >> 9` (512-byte aligned HBM address)
     - `+0x174`: `lhs_hbm >> 9`
     - `+0x190`: `out_hbm >> 9`
     - `+0x1e8`: `exec_seq` (`>= 1`)
3. **Doorbell Upload (`op=0x04`)**:
   - `dma_h2d_sync(host_stage_iova, 0x3fff00000ULL, 512, /*w28=*/0x0001fb10u, /*w29=*/0x01000000u, ...)`.
   - `w28 = 0x0001fb10` increments sync flag `0x1fb` by `16` units (`512 B / 32 B`), waking the resident core at `PC 18`.
4. **Completion Wait (`op=0x06`, `submit_exec_sync`)**:
   - Submit `d[1] = 0x06`, `d[8] = 0x41`, `d[10] = 0x010101f9` (sync flag `0x1f9`).
   - Assert completion queue words `cq[4] == 1u && cq[5] == exec_seq`.

---

## 2. Decoded 512-Bit VLIW ISA Primitives (`ot::VliwBundle`)

Each instruction bundle is 64 bytes (`16 x uint32_t`, little-endian):

| Primitive | Helper (`ot_vliw.h`) | Encoding |
| --- | --- | --- |
| Sync wait (`>= bytes/32`) | `make_sync_wait_imm(flag, bytes)` | `w[2]=0x3a8ca30f`, `w[12]=(flag & 0x3f) << 26`, `w[13]=((bytes/32) << 14) \| ((flag >> 6) & 7)` |
| Sync subtract (`-= bytes/32`) | `make_sync_sub_imm(flag, bytes)` | `w[2]=0x3a0ca10f`, `w[12]=(flag & 0x3f) << 26`, `w[13]=(((0 - bytes/32) & 0x3ffff) << 14) \| ((flag >> 6) & 7)`, `w[14]=0x0f000003` |
| Sync clear | `make_sync_clear(flag)` | `w[2]=0x3a0ca70f`, `w[12]=(flag & 0x3f) << 26`, `w[13]=0x00004000 \| ((flag >> 6) & 7)` |
| SMEM `ldesc` word load | `make_smem_load_ldesc(sreg, off)` | `neg_dwords = (int32_t(off) - 0x200) / 4`; `w[13]=uint32_t(neg_dwords) << 14`, `w[14]=0x0e041000 \| ((sreg & 0x1f) << 2)` |
| Scalar fence | `make_fence()` | `w[14]=0x0f000000`, `w[15]=0x55700900` |
| Absolute branch (4 delay slots) | `make_br_abs(target_pc)` | `w[13]=target_pc << 14`, `w[14]=0x0f000000`, `w[15]=0x55700400` |

### Predicate Encoding Note

Scalar slot `s1` (`w[14][28:24]`) and `s0` (`w[15][23:19]`) carry predicate selectors (`0x0e` = always true). At `PC 37`, replacing the guarded `!p2` launch-descriptor predicate (`w[14]=0x5201bee8`, `w[15]=0x55100dc3`) with unconditional `s1` (`w[14]=0x4e01bee8`, `w[15]=0x55780000`) removes the legacy `+0x1a8` branch guard so the LHS HBM -> VMEM DMA always executes unconditionally.

---

## 3. Exact 120-Bundle Resident IMEM Program Map

| Stage | PC Range | Count | Sync Flag(s) | Role |
| --- | --- | ---: | --- | --- |
| 1 | `PC 0..16` | 17 (`7` active, `10` NOP) | `0x1f9` | Load `SMEM[+0x1e8]` (`exec_seq`) and signal host completion `sync[0x1f9]` (`0` on cold boot, `exec_seq` after each matmul). |
| 2 | `PC 17..20` | 4 (`3` active, `1` NOP) | `0x1fb` | Block on host doorbell `sync[0x1fb] >= 16` (`512 B / 32 B`), then consume `sync[0x1fb] -= 16`. |
| 3 | `PC 21..27` | 7 (`7` active) | `0x1ff` | Issue 512 B HBM `0x3fff00000` -> SMEM DMA, `sync_wait(0x1ff, 512)`, `sync_sub(0x1ff, 512)`, and scalar `fence()`. |
| 4 | `PC 28..50` | 23 (`11` active, `12` NOP) | `0x33` | Load `s2 = out_hbm >> 9`, `s18 = lhs_hbm >> 9`, `s1 = rhs_hbm >> 9` (`6` latency NOPs), issue unconditional 32 KiB LHS HBM -> VMEM DMA, `sync_wait/sub/clear(0x33, 32768)`, and zero `s0`. |
| 5 | `PC 51..108` | 58 (`58` active) | `0x35`, `0x36` | Embedded `ot::emit_v5e_matmul_128x128_body()` (32 KiB RHS DMA, 16x 8-row MXU weight pushes, 4x 4 KiB activation loads, 8 MXU accumulate steps, 8 VSTORE flushes, 32 KiB output DMA, flag restore, fence). |
| 6 | `PC 109..119` | 11 (`1` active, `10` NOP) | — | `PC 109` NOP, `PC 110` `make_br_abs(0)` loop back to `PC 0`, `PC 111..114` 4 branch delay-slot NOPs, `PC 115..119` 5 alignment pad NOPs (`120` bundles = `7,680 B`). |

---

## 4. Causal Falsifications & Live Silicon Verification

Each synchronization stage in the 120-bundle loop was causally falsified on live `v5e-1` silicon:

- **Dropping Stage 2 (`sync[0x1fb]` wait at `PC 18..19`)**: core free-spins during host idle windows without doorbell synchronization.
- **Dropping Stage 3 (`sync[0x1ff]` wait/sub/fence at `PC 25..27`)**: `16,384 / 16,384` BF16 mismatches from racing launch-descriptor reads against HBM -> SMEM DMA.
- **Dropping Stage 4 (`sync[0x33]` wait/sub/clear at `PC 46..48`)**: intermittent `512..2,303 / 16,384` mismatches from LHS VMEM DMA race.
- **Unaligned IMEM DMA (`100` bundles = `6,400 B`)**: hardware timeout; `512 B`-aligned lengths (`104/112/120/128` bundles) pass.

### Stress & Numerical Parity Notes

- **100-Call Zero-Blob Stress**: 2 open/close cycles x 50 consecutive calls (`100` calls total), `0` CQ errors, average dispatch latency `133.1 us`, `MAPS_LIBTPU=0`, `MAPS_LIBPYTHON=0`.
- **Exact Dyadic BF16 Parity**: `50/50` random dyadic BF16 `128x128` `A @ B` runs (`[-2.0, +1.875]` step `0.125`, `819,200` total outputs) -> **`0 / 819,200` mismatches (`max_diff = 0.000000`)**.
- **Continuous Uniform BF16 Parity**: `50/50` continuous uniform `[-1, 1]` runs (including `lhs == rhs` aliasing and 512 B-aligned HBM addresses across `0x00100200..0x380000000`) -> `max_diff <= 0.015625` (`<= 1` BF16 ULP on exact reduction-tree tie boundaries).
- **Standalone End-to-End Driver Proof**: `libopentpu/tools/opentpu_matmul_proof.cpp` validates both C API `ot_matmul_bf16` (consecutive random + chained `(A @ B) @ D` runs) and `dlopen("./libopentpu.so")` -> `GetPjrtApi()` end-to-end execution with `0/16384` mismatches (`max_diff = 0.000000`), `MAPS_LIBTPU=0`, and `MAPS_LIBPYTHON=0`.
