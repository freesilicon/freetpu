# Tensor G5 (`buenos_a0`) Chained Multi-Op Fusion & `W8A16(BF16)` Scaling Specification

Clean-room ISA specification and C++17 porting blueprint for:
1. **Single-VII Multi-Op `0xa40` Chaining** (zero intermediate DRAM round-trips) decoded from live-verified Phase 6 probes (`fuse_fc64_silu_fc64`, `fuse_swiglu_64`, `fuse_qkv_64` — `3/3` `max_abs_err = 0` on `rango` bare-metal, `60.8–72.8 us` per 2–3 GEMM block).
2. **LLM-Scale Horizontal Fusion** (single-GEMM `[W_Q; W_K; W_V]` and `[W_gate; W_up]` requiring zero new opcodes).
3. **Native `W8A16(BF16)` FP32-I/O GEMM Scaling** (`probe_deq_b4a`/`b4b` live `32/32` `max_f32_err = 0.0` path).

Verified offline by [`tools/verify_chained_fusion.py`](../../tools/verify_chained_fusion.py).

---

## 1. Live `rango` Phase 6 Chained Fusion Corpus

All 3 models execute in **1 VII call** (`sgs = 1`) with **zero intermediate DRAM transfers** (all intermediate activations stay in on-chip SRAM) and pass `max_abs_err = 0` on `tpu_baremetal_runner`:

| Probe | Graph | Chunks | `ext[2]` (`pack_ext2` concat) | `0xa40` Channels | `pm5` HW (`rango`) | Live BM Parity |
|---|---|---:|---:|---|---:|---|
| `fuse_fc64_silu_fc64` | `FC1(64->64) -> SiLU -> FC2(64->64)` | `9` | `8,704 B` (`2 × 4,352 B`) | `0x01, 0x22, 0x42, 0x63` | `60.8 us` | `64/64` exact (`err=0`) |
| `fuse_swiglu_64` | `gate->SiLU * up -> down` (3 GEMMs) | `11` | `13,056 B` (`3 × 4,352 B`) | `0x01, 0x22, 0x42, 0x62, 0x83` | `65.4 us` | `64/64` exact (`err=0`) |
| `fuse_qkv_64` | `Q/K/V` 3-branch + `192 B` concat | `11` | `13,056 B` (`3 × 4,352 B`) | `0x01, 0x22, 0x42, 0x62, 0x83` | `72.8 us` | `192/192` exact (`err=0`) |

---

## 2. Exact Multi-Op `0xa40` Parameter-Stream Chaining Contract

### 2.1 `ext[2]` Layout (`8/8` Slices, `34,816 / 34,816 B` Bit-Exact)

`ext[2]` for an `S`-stage chained GEMM block is simply the **byte-concatenation** of each stage's standard `multitile_synth.pack_ext2(N_s, K_s, W_s, B_s, include_bias=True)` slice:

```text
ext[2] = [ stage_0_params (P_0 B) | stage_1_params (P_1 B) | ... | stage_{S-1}_params (P_{S-1} B) ]
```

For `64x64` INT8 + int32 bias: `P_s = 256 B (bias) + 4,096 B (weights) = 4,352 B = 0x1100 B`.

### 2.2 DIVE VM IR (`ext[0]`) Patching Pattern (`2*S + 5` Chunks Total)

For `S` chained GEMM stages (`S = 2` -> 9 chunks, `S = 3` -> 11 chunks):
- `chunk[0]` (`320 B`): Prolog (identical across all single- and multi-stage graphs).
- `chunk[1]` (`80 B`): Input DMA pre-setup (`0x680`, `0x940`).
- `chunk[2]` (`64 B`): Input `0xa40` DMA (`ch = 0x01`, `PutBits(chunk=2, offset=37, width=64)` <- `input_0` IOVA).
- For each stage `s = 0 .. S-1`:
  - `chunk[2*s + 3]`: Previous-stage completion / SRAM handoff + stage-`s` MXU/VPU CSR setup.
  - `chunk[2*s + 4]` (`64 B`): Stage-`s` parameter `0xa40` DMA:
    - Channel byte: **`ch = ((s + 1) << 5) | 0x02`** (`0x22, 0x42, 0x62, ...`).
    - DIVE VM IR: stage `0` uses `Copy(param_base)`; stages `s >= 1` emit **`Add(param_base, offset_s)`** (`offset_s = sum_{j<s} P_j`, e.g. `0x1100`, `0x2200`) -> **`PutBits(chunk=2*s+4, offset=37, width=64)`**.
- `chunk[2*S + 3]`: Final stage (`s = S-1`) GEMM execution + epilogue (`0x5c0`/`0x4c0`/`0x640`/`HDR_DATA` or QKV concat).
- `chunk[2*S + 4]` (`928 B` for 64B out, `1696 B` for 192B out): Output `0xa40` DMA (**`ch = ((S + 1) << 5) | 0x03`** = `0x63` for `S=2`, `0x83` for `S=3`) + `0x9c0` + gather + finish (`PutBits(chunk=2*S+4, offset=37, width=64)` <- `output_0` IOVA).

### 2.3 Parameter SRAM Ring Offset Stepping (`0x940` & `0x980`)

In stage `s`'s compute chunk (`chunk[2*s + 5]`, `s = 0, 1, 2`), the leading `0x940` (bias) and `0x980` (weight halves) step linearly through the parameter SRAM ring by **`stride_flits = P_s / 16`** (`0x110` for `4,352 B`):

| Packet | Field | Stage `s = 0` | Stage `s = 1` | Stage `s = 2` | Formula (`P = 4352 B`) |
|---|---|---:|---:|---:|---|
| `0x940` (flit 0) | `w1` | `0x10000` | `0x10110` | `0x10220` | `0x10000 + s * 0x110` |
| `0x940` (flit 2) | `w0` | `0xe0040311` | `0xe08c0311` | `0xe1140311` | `0xe0040311 + s * 0x880000` |
| `0x980` #1 (flit 3) | `w1` | `0x20000` | `0x20220` | `0x20440` | `0x20000 + s * 0x220` |
| `0x980` #1 (flit 5) | `w3` | `0x00202` | `0x11202` | `0x22202` | `0x00202 + s * 0x11000` |
| `0x980` #2 (flit 8) | `w1` | `0x20110` | `0x20330` | `0x20550` | `0x20110 + s * 0x220` |
| `0x980` #2 (flit 10) | `w3` | `0x08a02` | `0x19a02` | `0x2aa02` | `0x08a02 + s * 0x11000` |

Monotonic packet IDs (`pid`) increment sequentially across all chunks; `fc.c[0..4] == sw.c[0..4]` (`0` byte diffs) and `fc.c[7..8]` vs `sw.c[9..10]` differ **only** in `pid`, ring slot `s`, and `0xa40` output channel (`0x63 -> 0x83`).

---

## 3. In-SRAM Inter-Stage Operations (Zero DRAM Spill)

### 3.1 Sequential `FC1 -> SiLU -> FC2` (`fuse_fc64_silu_fc64 c[5]`)

1. `0x940` + `2×0x980` stream `W1, B1` and run MXU GEMM #1.
2. **`0x640` (flit 54, `tail_84b = 0x7804`) + `HDR_DATA` (flit 67)** run our exact Phase 5 Mode-A degree-4 `SiLU` spline directly in-situ on GEMM #1 output (`bps = [-6.192, -3.114, -0.717, 1.022, ...]`).
3. **`0x440` + `2×0x400`** scatter the `64 B` `SiLU(h1)` SRAM buffer directly back into MXU activation banks as the input for GEMM #2.
4. `0x200` CSR update reprograms MXU/VPU for GEMM #2 (`output_0`).

### 3.2 Full 3-GEMM SwiGLU (`fuse_swiglu_64 c[5], c[7]`)

1. `c[5]` (`2144 B`, `384 B` shorter than `fc64_silu_fc64 c[5]`): runs `gate` GEMM + **`0x640` (`tail = 0x7804`) `SiLU(gate)`** and parks `silu(gate)` in SRAM while **leaving `input_0` resident in MXU activation banks** (skipping the `0x440 + 2×0x400` re-scatter).
2. `c[7]` (`1888 B`): runs `up` GEMM, skips `0x640` completely, and executes **one 20-flit binary elementwise VPU packet `0x100` (flit 29, `w0 = 0x00003100`, tile mask `0x3`)** that multiplies the two resident SRAM tensors (`silu(gate) * up`) and requantizes in place -> `0x440 + 2×0x400` scatters `prod` into MXU activation banks -> `0x200` CSR update launches `down` (`c[9..10]`).

### 3.3 3-Branch Parallel `Q/K/V` + In-SRAM Concat (`fuse_qkv_64`)

- `c[5]` (`2016 B`) & `c[7]` (`2016 B`) are structurally identical (differing only in `pid`, param slot `s`, and destination SRAM bank `+0x000000` vs `+0x080000` for `0x440`/`0x400` and `0x408400/0x608400` vs `0x808400/0xa08400` for `0x6c0`).
- `c[9]` (`4800 B`) runs `V`, gathers `(Q, K, V)` via `0x600`/`0x540`, applies 6-tile `0x640` (`w0 = 0x0003f640`) + 12× `0x6c0` (tiles `0..5`, SRAM `0x00008000 .. 0x00a08400`) + 6-tile `HDR_DATA` (`w0 = 0x0003f200`), and `c[10]` (`1696 B`) emits 6× `0x400` into one `192 B` `0xa40` (`ch = 0x83`) output DMA.

---

## 4. Practical LLM Strategy for `libopenbuenos` (`SmolLM2-135M` & `Qwen2.5-1.5B`)

### 4.1 Immediate Zero-New-Opcode Win: Horizontal Single-GEMM Fusion (`QKV` & `gate+up`)

Because `Q/K/V` (and `gate`/`up`) share the exact same input activation vector `X` (`[M, K]`), concatenating their weight matrices along `N` into **one larger GEMM** (`[M, K] × [K, N_0 + N_1 + ...]`):
- **SmolLM2-135M**: `QKV` (`576 -> 576+192+192 = 960`) in **1 VII call** (instead of 3); `gate+up` (`576 -> 1536+1536 = 3072`) in **1 VII call** (instead of 2).
- **Qwen2.5-1.5B**: `QKV` (`1536 -> 1536+256+256 = 2048`) in **1 VII call** (saving `2 × ~330 us = 0.66 ms/layer = 18.5 ms/tok` on 28 layers); `gate+up` (`1536 -> 2×8960`) tiled by `N` with zero `0x600`/`0x540` concat packets.
- Works **today** with `libopenbuenos`'s existing single-GEMM synthesizer (`Q8_0`/`Q4_0` per-row dequant) and combines orthogonally with Mode-A `+0 B` `0x640` `SiLU` on the `gate` half.

### 4.2 Full Single-VII Multi-Op Chaining (When `ext[2]` Fits SRAM Streaming Budget)

For multi-layer blocks where intermediate activations fit on-chip (`M * N_mid <= SRAM budget`), emit `2*S + 5` chunks using §2–§3:
- Concat `ext[2] = [pack_ext2(W_0, B_0) | ... | pack_ext2(W_{S-1}, B_{S-1})]`.
- Emit DIVE `Add(param_base, offset_s)` + `PutBits(chunk=2*s+4, 37, 64)`.
- Step `0xa40` channels (`0x22, 0x42, 0x62, ... -> (S+1)<<5|0x03`) and `0x940`/`0x980` param ring offsets (`+P_s / 16` flits).
- Insert `0x640` (`tail=0x7804`) + `0x440/0x400` re-scatter for `FC1 -> SiLU -> FC2`, or `0x100` binary MUL for SwiGLU.

---

## 5. Native `W8A16(BF16)` FP32-I/O GEMM Scaling Spec (`b4a`/`b4b` Proven Path)

Phase 5 live silicon proved native hardware `W8A16(BF16)` FP32-I/O GEMM on `rango` (`probe_deq_b4a_u8_2048B`, `hw_us = 426`; `probe_deq_b4b_u8_2048B_c5_to_c7`, `hw_us = 411`, **`32/32` BIT-EXACT FP32 `max_f32_err = 0.0`**):

### 5.1 Hardware Dataflow & Exact Bit Contract

1. **Input (`FP32` DRAM -> `BF16` SRAM)**: `M × K × 4 B` raw IEEE-754 `float32` input (`0xa40 ch = 0x01`), rounded to `BF16` on-chip — **zero CPU activation quantization, zero outlier peeling**.
2. **Weights (`1 B/weight` `uint8` -> hardware `1:2` `BF16` unpack via `0x980 w8`)**:
   - Pack `W_u8[k, n] = clamp(round(W_f32[n, k] / s_n) + zp_w, 0, 255)` (for signed INT8 `w in [-128,127]`, `zp_w = 128` -> `u8 = w + 128`; for INT4 `w in [-8,7]`, `zp_w = 8` -> `u8 = w + 8`).
   - Layout in `ext[2][0 : K*N]`: **`[K // 2, N, 2]`** (pairs of consecutive `k` for each `n`, matching the BF16 MXU `2×1` pairing; exact size **`P_w = K * N` bytes** = identical DMA bandwidth to `Q8_0`).
   - **`0x980` word 8 (`w8`)**: **`(1u << 21) | ((uint32_t)zp_w << 22)`** (e.g. `0x02200000` for `zp_w = 8`, `0x20200000` for `zp_w = 128`). Hardware expands every `1 B` `uint8` into `2 B` `BF16(float(u8 - zp_w))` on the fly (`K*N` DRAM bytes -> `2*K*N` SRAM BF16 bytes).
3. **Per-Channel FP32 Scales (`ext[2][K*N : K*N + 4*N]`)**:
   - Raw IEEE-754 `float32[N]` appended at offset `K * N` (loaded by second parameter `0xa40 ch = 0x42`, `4 * N` bytes, DIVE `Add(param_base, K * N)`).
4. **BF16 Systolic MAC + Output (`FP32` DRAM)**:
   - Compute `CSR 0x56 = 0x26` (BF16 MXU mode), multiplies `BF16(A) × BF16(W_u8 - zp_w)` into FP32 accumulators, scales by per-channel `s_n`, and writes **`M × N × 4 B` raw IEEE-754 `float32`** directly to DRAM (`0xa40 ch = 0x63`).

### 5.2 Scaling Rules (`1x64x32` -> Larger `M × K × N` Tiles)

| Parameter | `b4a`/`b4b` (`1x64x32`) | General `M × K × N` Tile (`K` even, `N` mult of 32) |
|---|---|---|
| Input DRAM (`ch=0x01`) | `256 B` (`1×64×4`) | `4 * M * K` bytes (`FP32`) |
| Weight DRAM (`ch=0x22`, `ext[2]+0`) | `2,048 B` (`64×32×1`) | **`K * N` bytes** (`u8`, `[K/2, N, 2]` order; expands `2x` to `2*K*N` B BF16 in SRAM) |
| Scale DRAM (`ch=0x42`, `ext[2]+K*N`) | `128 B` (`32×4`) | `4 * N` bytes (`FP32`, DIVE `Add(param_base, K * N)`) |
| Output DRAM (`ch=0x63`) | `128 B` (`1×32×4`) | `4 * M * N` bytes (`FP32`) |
| `0x980 w8` | `0x02200000` (`zp=8`) | `(1u << 21) | (zp_w << 22)` (`zp_w = 128` for INT8 `Q8_0`, `zp_w = 8` for INT4 `Q4_0`) |
| **Bandwidth vs INT8 W8A8** | `1 B/weight` | **Same `1 B/weight` DMA BW**, `0` CPU quant/dequant/outlier-peel overhead, full BF16 activation dynamic range |

> **Recommendation for `libopenbuenos`**: Use `W8A8` (`Q8_0`/`Q4_0` + `perf_mode=5` + FW WakeLock) for maximum decode throughput (`27.6 GB/s`, `983.6 GOPS`), and bring up tiled `W8A16(BF16)` (`zp_w=128` / `zp_w=8`) specifically for outlier-heavy layers (e.g. `down_proj` / `lm_head` / `Qwen2.5` Layer 0–1) to eliminate CPU outlier-peeling overhead.
