# Tensor G5 (`buenos_a0`) Phase 7 Whole-Block In-SRAM Chained SwiGLU C++17 Specification

Clean-room ISA specification, closed-form C++17 emission tables, and `libopenbuenos` Milestone 7B integration blueprint for **single-VII (`sgs = 1`, `11` chunks) zero-DRAM-spill LLM SwiGLU** (`gate -> 0x640 SiLU * up [0x100 MUL] -> down`).

Verified offline against local reference corpora (gated behind `OPENBUENOS_CORPUS_DIR` in [`libopenbuenos/tests/openbuenos_host_test.cpp`](../../libopenbuenos/tests/openbuenos_host_test.cpp)):
- Hardware FP32-RNE Horner math parity: **`16/16` live `rango` probes, `0 / 181,760` byte diffs (`max_abs_err = 0`)**
- Closed-form parameter-ring parity: **`18/18` `0x940` + `184/184` `0x980` parameter-ring packets 100% bit-exact**
- Standalone non-PoT-scale `.dgc0` probes: **6 unseen LLM shapes** (including full `SmolLM2-135M` decode/prefill and `Qwen2.5-1.5B` decode/prefill).

---

## 1. Live Silicon Reference Corpus & `768 KiB` Narrow SRAM Ceiling

All **16/16** Phase 7 `rango` reference probes (`work/device-runner/phase7_perf/swiglu_refs/`) compiled into **1 single VII call (`sgs = 1`, `11` chunks)** — even with **`9.46 MiB` of streamed weights** — and matched our exact FP32-RNE Horner model with **`0 / 181,760` byte diffs (`max_abs_err = 0`)**:

| Probe (`M x K x H x N`) | `ext[2]` Bytes | `ext[2]` Format | `sgs / nchk` | `PM=5` `hw_us` (no FW WL)* | Est. w/ FW WL (`27.6 GB/s`) | Core-0 Activation SRAM | Exact HW Math Parity |
|---|---:|---|---:|---:|---:|---:|---:|
| `1x64x64x64_sdiff` | `13,056` | `3 x pack_ext2` | `1 / 11` | `64.0 us` | `~30 us` | `< 1 KiB` | `0 / 64` |
| `1x576x256x576` | `446,720` | `3 x pack_ext2` | `1 / 11` | `98.4 us` | `~45 us` | `< 2 KiB` | `0 / 576` |
| `1x576x512x576` | `891,136` | `3 x pack_ext2` | `1 / 11` | `126.6 us` | `~55 us` | `< 2 KiB` | `0 / 576` |
| **`1x576x1536x576` (`SmolLM2` FULL MLP)** | **`2,668,800`** | **`3 x pack_ext2`** | **`1 / 11`** | **`322.4 us`** | **`~110 us`** | **`< 4 KiB`** | **`0 / 576`** |
| `1x1536x256x1536` | `1,187,840` | `3 x pack_ext2` | `1 / 11` | `160.2 us` | `~65 us` | `< 4 KiB` | `0 / 1,536` |
| `1x1536x512x1536` | `2,369,536` | `3 x pack_ext2` | `1 / 11` | `223.4 us` | `~95 us` | `< 4 KiB` | `0 / 1,536` |
| **`1x1536x1024x1536` (`Qwen2.5` Decode)** | **`4,732,928`** | **`3 x pack_ext2`** | **`1 / 11`** | **`372.6 us`** | **`~175 us`** | **`< 4 KiB`** | **`0 / 1,536`** |
| `1x1536x2048x1536` | `9,459,712` | `3 x pack_ext2` | `1 / 11` | `905.8 us` | `~380 us` | `< 6 KiB` | `0 / 1,536` |
| `64x576x256x576` | `459,776` | `3 x pack_lt_64x` | `1 / 11` | `180.0 us` | `~105 us` | `68 KiB` | `0 / 36,864` |
| **`64x576x512x576` (`SmolLM2` Prefill)** | **`910,336`** | **`3 x pack_lt_64x`** | **`1 / 11`** | **`201.8 us`** (`561 GOPS`) | **`~120 us` (`~940 GOPS`)** | **`100 KiB`** | **`0 / 36,864`** |
| **`64x1536x512x1536` (`Qwen2.5` Prefill)** | **`2,400,256`** | **`3 x pack_lt_64x`** | **`1 / 11`** | **`427.8 us`** (`706 GOPS`) | **`~250 us` (`~1.2 TOPS`)** | **`160 KiB`** | **`0 / 98,304`** |

*\*Reference compiles ran via `tpu_direct_runner` without persistent FW WakeLock (`~12.5 GB/s`, matching Phase 6's `lt_1x2048x2048` `341 us` no-WL vs `152 us` with FW WL = `2.24x` speedup in `libopenbuenos`).*

### Exact `768 KiB` Narrow SRAM Ceiling Rules

1. **Weights do NOT reside in narrow activation SRAM**: weights stream continuously from DRAM through the **`256 KiB` (`0x40000`) hardware parameter ring** (`0x940` / `0x980`), supporting arbitrary multi-MiB `ext[2]` (`9.46 MiB` verified in `sgs=1`).
2. **Activation SRAM residency (`BUENOS_CORE_NARROW_MEM_BYTES = 786,432 B`)**:
   - **`M = 1` (16-core / 9-core distributed)**: per-core narrow activation residency is only `(K + 2*H) / cores` (`<= 4 KiB`), so `H <= 1024` (`1x1536x1024x1536`, `0.364 us/H` optimal!) and `H = 1536` (`1x576x1536x576`, **100% of SmolLM2 MLP in 1 VII call!**) stay far below SRAM limits. At `H = 2048` (`1x1536x2048x1536`), `c[8]` expands from `64 B` to `1,952 B` and per-`H` latency degrades (`0.442 us/H` vs `0.364 us/H`), making **`H = 1024`** the optimal `Qwen2.5-1.5B` decode chunk size.
   - **`M = 64` (core-0 gather-compute-scatter)**: core 0 simultaneously holds `input_0` (`64*K`), `gate/silu/prod` (`64*H`, overwritten in place!), and `up` (`64*H`), then overwrites dead `input_0` with `output_0` (`64*N`):
     ```text
     Peak Core-0 Narrow Activation SRAM (M=64) = 64 * (max(K, N) + 2 * H) bytes
       - 64x576x512x576  : 64 * (576 + 1024)  = 102,400 B (100 KiB << 768 KiB)
       - 64x1536x512x1536: 64 * (1536 + 1024) = 163,840 B (160 KiB << 768 KiB)
     ```
     Both production `M=64` prefill shapes fit with `> 600 KiB` headroom!

---

## 2. Clean-Room Container, `ext[2]`, and 11-Chunk DIVE VM IR (`ext[0]`)

### 2.1 `ext[2]` Parameter Layout (`16/16` Bit-Exact)

Simply concatenate the 3 stages using `ob_gemm_emitter.cpp`'s **existing** packers:
```cpp
// M == 1:
ext2 = pack_ext2(H, K, W_gate, B_gate, true)      // P_gate = H * K + 4 * H
     + pack_ext2(H, K, W_up,   B_up,   true)      // P_up   = H * K + 4 * H
     + pack_ext2(N, H, W_down, B_down, true);     // P_down = N * H + 4 * N

// M == 64:
ext2 = pack_ext2_lt_64x_impl(H, K, W_gate, B_gate) // P_gate = H * (K + 16)
     + pack_ext2_lt_64x_impl(H, K, W_up,   B_up)   // P_up   = H * (K + 16)
     + pack_ext2_lt_64x_impl(N, H, W_down, B_down);// P_down = N * (H + 16)
```

### 2.2 11-Chunk DIVE VM IR (`build_dive_ext0_11chunk`)

Extend `build_dive_ext0_clean` (`ob_gemm_emitter.cpp:2458`) from `7` chunks (`14` values, `24` ops) to **`11` chunks (`16` values, `32` ops)**:
- Add two values: `val[14] = "variable_5.3"`, `val[15] = "variable_5.4"` (`dt = 11`, empty dims).
- Add detail kind `DET_ADD`: `fb.start_table(); fb.add_u32(0u, byte_offset, 0u, /*force=*/true); det_off = fb.end_table();`
- Stage DMA ops (`PutBits(offset = 37, bit_width = 64)`):
  1. `input_0` (`ch[1..2]`): `Copy(8 -> 11)`, `Dispatch(1)`, `PutBits(ch=2, in=11)`, `Dispatch(2)` — `0xa40 ch = 0x01`, `M*K` B
  2. `gate` (`ch[3..4]`): `Copy(10 -> 12)`, `Dispatch(3)`, `PutBits(ch=4, in=12)`, `Dispatch(4)` — `0xa40 ch = 0x22`, `P_gate` B
  3. `up` (`ch[5..6]`): **`Add(op_type=3, imm=P_gate, in=10 -> out=13)`**, `Dispatch(5)`, `PutBits(ch=6, in=13)`, `Dispatch(6)` — `0xa40 ch = 0x42`, `P_up` B
  4. `down` (`ch[7..8]`): **`Add(op_type=3, imm=P_gate+P_up, in=10 -> out=14)`**, `Dispatch(7)`, `PutBits(ch=8, in=14)`, `Dispatch(8)` — `0xa40 ch = 0x62`, `P_down` B
  5. `output_0` (`ch[9..10]`): `Copy(9 -> 15)`, `Dispatch(9)`, `PutBits(ch=10, in=15)`, `Dispatch(10)` — `0xa40 ch = 0x83`, `M*N` B
- `dive_scratch` and `MultiExecutable.field_14` (hardware narrow-SRAM reservation — **REQUIRED**: under-sizing `field_14` causes firmware VII `code=10`!):
  ```cpp
  uint32_t scratch_sz = 196608u + (M == 64u ? 2u * align_up_u32(M * K, 4096u) : 0u);
  // M=1 -> 196,608 B; 64x576 -> 270,336 B; 64x1536 -> 393,216 B

  uint32_t exec_field_14 = (M == 64u)
      ? (1024u * (std::max(K, N) + 2u * H) + 320u)  // 64x576x512 -> 1,638,720; 64x1536x512 -> 2,621,760
      : 32768u;                                      // M=1 max needed <= 27,904
  ```

---

## 3. Exact Multi-Stage Parameter Ring Formulas (`0x940` & `0x980`)

Verified **18/18 (`0x940`)** and **184/184 (`0x980`)** bit-exact across `4,352 B .. 9,459,712 B`!

Let `off` be the cumulative byte offset into `ext[2]` (`0` for `gate`, `P_gate` for `up`, `P_gate + P_up` for `down`) and `P` be the stage byte size (`P_gate`, `P_up`, `P_down`). The hardware parameter ring wraps modulo **`0x40000` (`256 KiB`)** with **`128 KiB` (`0x20000`)** inner TTU loop span:

```cpp
// 48-byte (3-flit) 0x940 parameter ring producer for stage (off, P).
// Note: is_big_1536 is true ONLY on 1x576x1536x576 (w4 low16 = 0x0310 vs 0x0311).
Pkt pkt_940_params_ring(uint64_t off, uint64_t P, bool is_big_1536 = false) {
  const uint64_t sram_off16 = (off % 0x40000ULL) >> 4;
  const uint64_t w0 = (1ULL << 48) | (sram_off16 << 32) | 0x940ULL;
  const uint64_t d1 = P / 0x20000ULL;
  const uint64_t w1 = 0x3C00400078008FFFULL | (d1 << 31);
  const uint64_t w2 = 0x0F0010001E002000ULL;
  const uint64_t rem = P % 0x20000ULL;
  uint64_t w3_hi = 0ULL;
  if (rem != 0ULL) {
    const uint64_t slack = (0x1000ULL - (rem >> 4)) & 0xFFFULL;
    const uint64_t slots_m1 = (rem >> 5) - 1ULL;
    w3_hi = (slack << 20) | (1ULL << 18) | (slots_m1 << 3);
  }
  const uint64_t w3 = (w3_hi << 32) | 0x80010800ULL;
  const uint64_t w4_hi = (off < 0x40000ULL)
      ? (0x3FFE004ULL + (off >> 5))
      : (0x2000004ULL + ((off - 0x40000ULL) >> 5));
  const uint64_t w4 = (w4_hi << 16) | (is_big_1536 ? 0x0310ULL : 0x0311ULL);
  return {{w0, w1}, {w2, w3}, {w4, 4ULL}};
}

// 80-byte (5-flit) 0x980 per-core parameter ring consumer for core g in [0..n_cores-1].
// For M=64, n_cores == 1 (g == 0, grp_bytes == P, w6..w8 use core-0 bp_64x flags).
Pkt pkt_980_params_ring(uint64_t off, uint32_t g, uint64_t grp_bytes, bool is_m64 = false) {
  const uint64_t byte_off = off + static_cast<uint64_t>(g) * grp_bytes;
  const uint64_t sram_off8 = (byte_off % 0x40000ULL) >> 3;
  const uint64_t w0 = (1ULL << 49) | (sram_off8 << 32) | 0x980ULL;
  const uint64_t d1 = grp_bytes / 0x20000ULL;
  const uint64_t w1 = 0x78008000F0011FFEULL | (d1 << 32);
  const uint64_t w2 = 0x1E0020003C004000ULL;
  const uint64_t rem = grp_bytes % 0x20000ULL;
  uint64_t w3_hi = 1ULL, w4 = 0ULL;
  if (rem != 0ULL) {
    const uint64_t slack13 = (0x2000ULL - (rem >> 4)) & 0x1FFFULL; // 13-bit (128 KiB budget)
    const uint64_t slots_m1 = (rem >> 5) - 1ULL;
    const uint64_t val34 = (slack13 << 21) | (1ULL << 19) | (slots_m1 << 4) | 1ULL;
    w3_hi = val34 & 0xFFFFFFFFULL;
    w4 = (val34 >> 32) & 0xFULL;
  }
  const uint64_t w3 = (w3_hi << 32) | 0x00021000ULL;
  const uint64_t abs64 = byte_off >> 6;
  const uint64_t w5 = (abs64 << 42) | 0x0000020221FF8C00ULL;
  const uint32_t tb = kSnake16[g];
  const uint64_t w6 = is_m64 ? /* see bp_64x w6 */ 0x0000004458000004ULL : 4ULL;
  const uint64_t w7 = is_m64 ? 0x0181000000C00000ULL
                             : (0x0001000000800000ULL | (tb < 8u ? (1ULL << (56u + tb)) : 0ULL));
  const uint64_t w8 = (!is_m64 && tb >= 8u) ? (1ULL << (tb - 8u)) : 0ULL;
  return {{w0, w1}, {w2, w3}, {w4, w5}, {w6, w7}, {w8, 0ULL}};
}
```
*(For `M=64`, `0x940` is placed at the **tail** of `c[3], c[5], c[7]` and `0x980` is placed at the **head** of `c[5], c[7], c[9]`; for `M=1`, both `0x940` and `n_cores x 0x980` are at the **head** of `c[5], c[7], c[9]`.)*

---

## 4. Exact Hardware Math & Scale Encoding (`0x640` SiLU + `0x100` MUL + `OP20` / `CSR`)

### 4.1 Exact Hardware Dataflow (`0 / 181,760` Diffs on `rango`)

All intermediate tensors (`gate`, `up`, `silu`, `prod`) remain in internal unsigned `uint8 = int8 + 128` (`zp_hw = 128`) SRAM representation with zero intermediate MSB-flips:
1. **`gate` GEMM (`c[3]` `CSR_0x040`)**: `q_g = clamp(RNE(acc_g * M_gate), -128, 127)` where `M_gate = (s_in * s_wg) / s_gate`.
2. **`up` GEMM (`c[5]` `OP20_0x040` or `CSR_0x040`)**: `q_u = clamp(RNE(acc_u * M_up), -128, 127)` where `M_up = (s_in * s_wu) / s_up`.
3. **`SiLU` (`0x640` `tail=0x7804` + `OP20_0x200`)**:
   - `OP20_0x200` dequantizes `u_g = q_g + 128`: `x = clamp((u_g - 128) * s_gate, -11.6f, +inf)`.
   - `0x640` evaluates the 8-segment degree-4 Horner spline `y = P_seg(x) / s_silu`, then adds `zp_out_hw = 128` and clamps `u_silu = clamp(RNE(y) + 128, 0, 255)`.
4. **Binary Elementwise `MUL` (`0x100` `OP20_0x100` or `CSR_0x100`, `0x56 = 0x100e00`)**:
   - Reads `u_silu` (`zp0 = 128`) and `u_up` (`zp1 = 128`) directly from SRAM:
   - `q_prod = clamp(RNE(((u_silu - 128) * (u_up - 128)) * M_mul), -128, 127)` where **`M_mul = (s_silu * s_up) / s_prod`**, stored as `u_prod = q_prod + 128`.
5. **`down` GEMM (`c[7]` `CSR_0x040`)**: `q_out = clamp(RNE((q_prod @ W_down^T + B_down) * M_down), -128, 127)` where `M_down = (s_prod * s_wd) / s_out`.
6. **Final MSB-Flip (`c[9]` `0x640` `tail=0x7801` + `OP20_0x200`)**: flips `u_out = q_out + 128` -> signed `int8` `q_out` for DRAM output `0xa40 ch = 0x83`.

### 4.2 Universal Degree-4 `0x640` SiLU Coefficient Table (`s_silu = 1.0`)

Verified bit-identical across **16/16** live `rango` probes. For **any** runtime `s_silu`, emit `coeffs[s][k] = kSiluUnitCoeffs[s][k] / s_silu` with `tail_84b = 0x7804`:

```cpp
static constexpr float kSiluBreakpoints[7] = {
    -6.1918535232543945f, -3.113853931427002f, -0.7166378498077393f,
     1.0220223665237427f,  3.214235782623291f,  6.553915500640869f,
    12.300000190734863f
};

static constexpr float kSiluUnitCoeffs[8][5] = {
    {-0.370022177696228f,  -0.14004893600940704f, -0.02006860077381134f, -0.0012858827831223607f, -3.100011235801503e-05f},
    {-0.6562100052833557f, -0.26750609278678894f, -0.03700931742787361f, -0.001571062603034079f,   2.259612301713787e-05f},
    { 0.018058594316244125f, 0.5728806853294373f,  0.363774836063385f,    0.0851035863161087f,     0.007186070084571838f},
    { 5.7220458984375e-05f,  0.5000313520431519f,  0.2489282339811325f,  -0.0001401097106281668f, -0.017839739099144936f},
    { 0.01684805378317833f,  0.42960694432258606f, 0.36192697286605835f, -0.08451402187347412f,    0.0071176812052726746f},
    {-0.6919266581535339f,   1.2996355295181274f, -0.04768627509474754f,  0.0031246759463101625f, -6.095305070630275e-05f},
    {-0.30064311623573303f,  1.1088173389434814f, -0.014867686666548252f, 0.0009060873999260366f, -2.0734756617457606e-05f},
    { 0.0f,                  1.0f,                 0.0f,                  0.0f,                    0.0f}
};
```

### 4.3 `OP20` (`0x040`/`0x100`/`0x200`) vs `CSR` (`0x040`/`0x100`) Scale Rules

1. **Every 20-flit `OP20` packet (`0x040`, `0x100`, `0x200`)** packs the entire 98-register `CsrBank[0..0x61]` linearly at **bit offset `77 + 25 * reg`** (width `25` bits) with **`tail33 == 0` (bits `[2559:2527] == 0`)** verified across all 6 shapes — a 5-line drop-in C++17 builder:
   ```cpp
   Pkt build_op20_from_csrs(uint32_t op, uint32_t mask, bool ef50, const CsrBank& csrs) {
     std::vector<uint8_t> raw(320u, 0u);
     uint64_t lo0 = (static_cast<uint64_t>(mask & 0xFFFFu) << 12) | (op & 0xFFFu);
     uint64_t hi0 = ef50 ? 0x50ULL : 0ULL;
     std::memcpy(raw.data(), &lo0, 8u);
     std::memcpy(raw.data() + 8u, &hi0, 8u);
     for (uint32_t r = 0u; r < 0x62u; ++r) {
       set_bits_le(raw.data(), raw.size(), 77u + 25u * r, 25u, csrs[r]);
     }
     return bytes_to_flits(raw);
   }
   ```
   In particular, `f17..f19` (`b17 = 17 * 128 = 2176`) contains:
   - `b17 + 64` (`8b`, `0x56[20:13]` / `0x57` LHS zp): `zp0_hw` (`128` for `OP20_0x040` / `OP20_0x100`)
   - `b17 + 80` (`8b`, `0x57[11:4]`): `zp1_hw` (`128`)
   - `b17 + 189` (`32b`, `0x5B[24:13] | 0x5C[19:0]`): **`fp32(scale)`** (`M_up` in `OP20_0x040`, `s_gate` in SiLU `OP20_0x200`, `M_mul` in `OP20_0x100`)
   - `b17 + 222` (`32b`): **`fp32(fclip_max)`** (`127.0f` for GEMM/MUL, `+inf` for SiLU)
   - `b17 + 259` (`32b`): **`fp32(fclip_min)`** (`-128.0f` for GEMM/MUL, `-11.6f` for SiLU)
   - `b17 + 328` (`8b`, `0x61[9:2]`): `zp_out_hw` (`128`)
2. **CRITICAL Shadow-Register Safety Rule for `CSR_0x100` & `CSR_0x040`**:
   - In compiler-emitted `CSR_0x100`, `0x5C` is omitted whenever `M_mul == s_gate`, and in `c[7]` `CSR_0x040` (`down`), `0x5B` is omitted whenever `M_mul` and `M_down` both have `lo12 == 0`.
   - **Always emit ALL FIVE registers `0x5B, 0x5C, 0x5D, 0x5E, 0x5F` explicitly** (using `ob_gemm_emitter.cpp:1127-1131`) in every `CSR_0x040` and every main `CSR_0x100` packet! This guarantees bit-exact operation for **arbitrary distinct non-PoT scales** (`M_gate != M_up != s_gate != s_silu != M_mul != M_down`)!
   - Preserve the `CSR_0x100` header low byte `ef_lo`: `0x000` for `M=1 H=512`, **`0x070`** for `M=1 H=1024` and `M=64 H=512` (bit `0x20` enables dual-input `0x6c0` streams).

---

## 5. Exact Chunk-by-Chunk C++17 Emission Blueprint (`ob_gemm_emitter.cpp`)

### 5.1 `M = 1` Large-Tile SwiGLU (`synthesize_swiglu_1x_chunks`)

Supports **`(1, 576, 1536, 576)` (`SmolLM2` full MLP)**, **`(1, 1536, 1024, 1536)` (`Qwen2.5` decode)**, **`(1, 1536, 512, 1536)`**, and `(1, 576, 512, 576)`. Reuses `synthesize_lt_1x_chunks` primitives directly:

| Chunk | Role | Exact Composition from Existing `ob_gemm_emitter.cpp` Helpers |
|---|---|---|
| `c[0..2]` | Prolog + `input_0` | Identical to `build_chunk0()`, `build_chunk1(1, K)`, `build_chunk2(1, K)`. |
| `c[3]` | `input_0` load + `gate` setup | **Identical to `lt_1x(K, H)` `c[3]`** (with `M_gate` in `CSR_0x040`). |
| `c[4]` | `W_gate` DMA | `0xa40(ch = 0x22, nbytes = P_gate)`. |
| `c[5]` | `gate` MAC + (`H<=1024`: `up` launch; `H=1536`: `SiLU` + `up` launch) | `pkt_940_params_ring(0, P_gate)` + `h_cores x pkt_980_params_ring(0, g, P_gate/h_cores)` + `5` sync pkts. **If `H <= 1024`**: repeat `gate`'s `h_cores x 0x440` + `0x500` + `OP20_0x040(tgt_up)` where `tgt_up == tgt_gate` except **`tgt_up[0x17] += 1`**, `tgt_up[0x46] &= 0xFF`, and `M_up` in `0x5B..0x5C`! **If `H == 1536`**: emit `0x5c0/0x4c0/0x680/0x640(SiLU)/OP20_0x200(s_gate, -11.6, +inf)/CSR_0x200(iters3)` + repeat `16 x 0x440` + `0x500` + `CSR_0x040(up)`! |
| `c[6]` | `W_up` DMA | `0xa40(ch = 0x42, nbytes = P_up)`. |
| `c[7]` | `up` MAC + (`H<=1024`: `SiLU` + `CSR_0x100`; `H=1536`: `OP20_0x100`) + `prod` bcast + `down` setup | `pkt_940_params_ring(P_gate, P_up)` + `h_cores x pkt_980_params_ring(P_gate, g, P_up/h_cores)` + `5` sync pkts. **If `H <= 1024`**: `0x5c0/0x4c0/0x680/0x640(SiLU)` + (`H=1024`: `2x 0x6c0` at `sram_base + 32*p`) + `OP20_0x200(s_gate)` + (`H=1024`: `3x 0x6c0` LHS=`sram_base+32*p`, OUT=`sram_base+32*p`, RHS=`sram_base+128+32*p`) + **`CSR_0x100(M_mul)`** (table below!). **If `H == 1536`**: **`OP20_0x100(M_mul)` + `CSR_0x100(iters3)`** (table below!). Then append **`lt_1x(H, N)` `c[3]` broadcast + `n_cores x 0x440` + `0x500` + `CSR_0x040(down, M_down)`**! |
| `c[8]` | `W_down` DMA | `0xa40(ch = 0x62, nbytes = P_down)`. |
| `c[9]` | `down` MAC + MSB-flip | `pkt_940_params_ring(P_gate+P_up, P_down)` + `n_cores x pkt_980_params_ring(P_gate+P_up, g, P_down/n_cores)` + **identical `lt_1x(H, N)` `c[5]` tail**! |
| `c[10]` | `output_0` DMA + finish | **Identical `lt_1x(H, N)` `c[6]`** with `0xa40 ch = 0x83` and `tok += c3_stage0_ch1_tokens` (`+5` on `1x576x1536x576`, `+0` when `k_cores == h_cores == n_cores == 16`)! |

#### Exact `CSR_0x100` Binary MUL Tables (`M = 1`, `mask = 0xFFFF`)
- **`H = 512` (`ef_lo = 0x000`, `sw_unit = ceil(K/128)`)**:
  - `0x00=0x4000, 0x06=0x100008, 0x07=0x1f80000, 0x08=0x7ff, 0x09=0x400, 0x0a=0, 0x0c=0x20000, 0x0d=0, 0x10=0, 0x13=0x154, 0x16=0x3fc08`, **`0x17 = 0x2000 | sw_unit`**, `0x18..0x24=0, 0x27=0x1000000, 0x28=0xf8000f`, **`0x29 = (sw_unit + 1) << 21`**, `0x2a=0x2000, 0x2c=0x1ffff0, 0x2d=0x1f00000, 0x2e=0xfff, 0x2f=0x1fff800, 0x30=7, 0x31=0x7fffc, 0x32=0x1fc0000, 0x33=0x3ff, 0x34=0x1fffe00, 0x35=1, 0x36=0x1ffff, 0x37=0x1000, 0x43=0x20000, 0x45..0x48=0, 0x56=0x100e00, 0x58=0x4000` + explicit **`0x5B..0x5F = encode_scale_clip_csrs(M_mul, 127.0f, -128.0f)`**.
- **`H = 1024` (`ef_lo = 0x070`, `K`-independent!)**:
  - Identical to `H = 512` except `0x09=0xc00, 0x0c=0x60000, 0x10=1, 0x17=0x2000` and omit `0x29` (the `3x 0x6c0` packets carry `sram_base` & `sram_base + 128`).
- **`H = 1536` (`1x576x1536x576`, `OP20_0x100` + 7-flit `CSR_0x100(iters3)`)**:
  - `OP20_0x100` (`hi0 = 0x00000000_18000000`, `mask = 0xFFFF`): `0x00=0xc000, 0x05=0x2800, 0x06=0x300008, 0x07=0x1e80000, 0x08=0x7ff, 0x09=0xc00, 0x0b=0x3fffa, 0x0c=0x60000, 0x0e=0x1fffd00, 0x0f=0x1000000, 0x10=1, 0x11=0x1fe8000, 0x12=0x7f, 0x13=0x154, 0x16=0x3fc08, 0x17=0x200c, 0x27=0x1000000, 0x28=0x1f8001f, 0x29=0x1800000, 0x2a=0x2000, 0x2b=0x60, 0x2c=0x1fffd0, 0x2d=0x1d00000, 0x2e=0xfff, 0x2f=0x1ffe800, 0x30=7, 0x31=0x7fff4, 0x32=0x1f40000, 0x33=0x3ff, 0x34=0x1fffa00, 0x35=1, 0x36=0x1fffd, 0x37=0x1000, 0x3b=0x3000, 0x43=0x20000, 0x55=0x85000, 0x56=0x100e00, 0x57=0x800, 0x58=0x4000, 0x59=0x1ef, 0x61=0x200` + `0x5B..0x5F` (`M_mul`, `127.0f`, `-128.0f`), followed by 7-flit `CSR_0x100` (`0x05=0x2c00, 0x06=0x100008, 0x07=0x1f80000, 0x09=0x1400, 0x0b=0x3fffe, 0x0c=0xa0000, 0x0e=0x1ffff00, 0x10=2, 0x11=0x1ff8000, 0x16=0x103fc08, 0x28=0xf8000f, 0x29=0x1900000, 0x2b=0x20, 0x2c=0x1ffff0, 0x2d=0x1f00000, 0x2f=0x1fff800, 0x31=0x7fffc, 0x32=0x1fc0000, 0x34=0x1fffe00, 0x36=0x1ffff, 0x3b=0x1000, 0x55=0x105000`).

### 5.2 `M = 64` Batched Prefill SwiGLU (`synthesize_swiglu_64x_chunks`)

Supports **`(64, 576, 512, 576)` (`SmolLM2` exact `3x512`)** and **`(64, 1536, 512, 1536)` (`Qwen2.5` prefill)**! Remarkably compact: `c[5] = 960 B` (`11` pkts), `c[7] = 2,576 B` (`32` pkts), `c[9] = 3,440 B`, `c[10] = 3,616 B`:

- **Core-0 Narrow SRAM Map (`H = 512`, `64*H = 0x8000`, `16*H = 0x2000`)**:
  - `K = 576`: `gate/silu/prod = 0x000000`, `input_0 / output_0 = 0x008000` (`32 KiB`, `16*K = 0x2400`), `up = 0x011000` (`68 KiB`).
  - `K = 1536`: `input_0 / output_0 = 0x000000` (`16*K = 0x6000`), `gate/silu/prod = 0x018000` (`96 KiB`), `up = 0x020000` (`128 KiB`).
- **`c[0..3]`**: Identical to `synthesize_lt_64x_chunks(K, H)` with `in_sram_base` from the SRAM map above, `gate` output at `gate_base`, and `M_gate` in `CSR_0x040`, ending with `pkt_940_params_ring(0, P_gate)`.
- **`c[5]` (`11` pkts, `960 B`)**: `pkt_980_params_ring(0, 0, P_gate, true)` -> `0x680(3f9)` -> `0x900(bar_all)` -> repeat `0x440` & `0x500` -> `4x 0x6c0` (reads `input_0` at `in_sram_base + 16*K*j`, writes `up` at `up_base + 16*H*j`) -> **`OP20_0x040(up, M_up)`** -> `pkt_940_params_ring(P_gate, P_up)`.
- **`c[7]` (`32` pkts, `2,576 B`)**: `pkt_980_params_ring(P_gate, 0, P_up, true)` -> `0x680` -> `build_op5c0_m64(H)` -> `build_op4c0_m64(H)` -> `0x680(3e0)` -> `0x640(SiLU, s_silu)` -> `4x 0x6c0` (in-place at `gate_base + 16*H*j`) -> `OP20_0x200(s_gate, -11.6, +inf)` -> `2x 0x680` -> `6x 0x6c0` (LHS=`gate_base+16*H*j`, OUT=`gate_base+16*H*j`, RHS=`up_base+16*H*j`) -> **`CSR_0x100(M_mul)`** (100% identical 56-CSR table for BOTH `K=576` and `K=1536`, `mask = 0x0001, ef_lo = 0x070`, explicit `0x5B..0x5F`!) -> `3x 0x680` -> `0x900` -> `down`'s `0x440` & `0x500` -> `4x 0x6c0` (reads `prod` at `gate_base + 16*H*j`, writes `output_0` at `in_sram_base + 16*N*j`) -> `CSR_0x040(down, M_down)` -> `pkt_940_params_ring(P_gate+P_up, P_down)`.
- **`c[9..10]`**: Identical to `synthesize_lt_64x_chunks(H, N)`'s `c[5..6]` with `0x980` at `off = P_gate + P_up`, `output_0` at `in_sram_base`, and `0xa40 ch = 0x83`!

#### Exact `CSR_0x100` Binary MUL Table (`M = 64, H = 512`, `mask = 0x0001, ef_lo = 0x070`, `K`-Independent!)
Identical for BOTH `64x576x512x576` and `64x1536x512x1536` (append explicit `0x5B..0x5F = encode_scale_clip_csrs(M_mul, 127.0f, -128.0f)`):
```cpp
static constexpr std::pair<uint32_t, uint32_t> kM64H512MulCsrs[] = {
    {0x00, 0x1c000}, {0x02, 0xfc0},     {0x06, 0x700008}, {0x07, 0x1c80000},
    {0x08, 0x7ff},   {0x09, 0x6400},    {0x0a, 0x0},      {0x0b, 0x3fff2},
    {0x0c, 0x320000},{0x0d, 0x0},       {0x0e, 0x1fff900},{0x10, 0x3f000c},
    {0x11, 0xc8000}, {0x12, 0x7e},      {0x13, 0x154},    {0x18, 0x7e00},
    {0x19, 0x1f82000},{0x1a, 0x0},      {0x1b, 0xfc10},   {0x1c, 0x100000},
    {0x1d, 0x7e},    {0x1e, 0x7e0800},  {0x1f, 0x0},      {0x20, 0x3f04},
    {0x21, 0x1040000},{0x23, 0x1f8200}, {0x24, 0x10000},  {0x27, 0x1000000},
    {0x28, 0x1f8003f},{0x29, 0x1},      {0x2a, 0x2000},   {0x2b, 0xe0},
    {0x2c, 0x1e00190},{0x2d, 0x1900003},{0x2e, 0xfc0},    {0x2f, 0x1c0c800},
    {0x30, 0x7},     {0x31, 0x7e064},   {0x32, 0x640000}, {0x33, 0x3f0},
    {0x34, 0x1f03200},{0x35, 0x1},      {0x36, 0x1f819},  {0x37, 0x3000},
    {0x3b, 0x1ff000},{0x3c, 0x0},       {0x43, 0x20000},  {0x45, 0x0},
    {0x46, 0x0},     {0x47, 0x0},       {0x48, 0x0},      {0x56, 0x100e00},
}; // + {0x5B..0x5F} from encode_scale_clip_csrs(M_mul, 127.0f, -128.0f)
```

---

## 6. `libopenbuenos` Milestone 7B Integration Blueprint (`ob_gemm_swiglu_f32_quant`)

### 6.1 Exact Per-Row Quantization Algebra (Zero Accuracy Loss on `up` & `down`)

Real LLMs (`SmolLM2-135M`, `Qwen2.5-1.5B`) store `W_gate`, `W_up`, `W_down` in `Q8_0`/`Q4_0` with zero MLP bias. To run the entire SwiGLU block in 1 VII call while preserving **full per-row precision** on `up` and `down`:
1. **`W_up[h, k]`**: quantize per-row to `int8 W_up_q[h, k]` with exact per-row scale `row_s_u[h]`.
2. **Absorb `row_s_u[h]` into `W_down` at weight-pack time (1-time LRU cache fill!)**:
   ```cpp
   // Exact algebraic identity:
   //   sum_h SiLU(gate[h]) * (up_q[h] * row_s_u[h]) * W_down[n, h]
   // = sum_h (SiLU(gate[h]) * up_q[h]) * (row_s_u[h] * W_down[n, h])
   W_down_eff[n, h] = W_down[n, h] * row_s_u[h];
   ```
   Quantize `W_down_eff[n, h]` **per-row** (`n = 0..N-1`) to `int8 W_down_q[n, h]` with exact per-row scale `row_s_d[n]`.
3. **`W_gate[h, k]`**: quantize per-chunk (`H_chunk`) to `int8 W_gate_q[h, k]` with common chunk scale `s_wg = max(|W_gate|) / 127.0f` (needed because `SiLU` is nonlinear).
4. **Runtime scales & guaranteed zero-overflow `SiLU`**:
   - Quantize `X[m, k]` to `int8 X_q` with scale `s_a` (rounded up to a `2^(1/4)` grid so `s_gate` stays in ~4–8 cache buckets across all tokens!).
   - Choose `D_gate`, `D_up`, `D_down` via `compute_tile_scale_D(...)` and `D_mul = 64` (or `128`).
   - Set **`s_gate = s_a * s_wg * D_gate`** and **`s_silu = s_gate`** (guarantees `SiLU(x) / s_silu in [-128, 127]` for all `s_gate >= 0.0022f`!).
   - Program hardware multipliers: `M_gate = 1/D_gate`, `M_up = 1/D_up`, `M_mul = 1/D_mul`, `M_down = 1/D_down`.
   - Final CPU FP32 dequant (NEON):
     ```cpp
     // For each output n = 0..N-1 (additive across H-chunks!):
     Y_f32[m, n] += (float)q_out[m, n] * (s_silu * s_a * D_up * D_mul * D_down * row_s_d[n]);
     ```

### 6.2 Additive `H`-Chunking & Graph/Weight Caching

Because `SwiGLU(X, W_g, W_u, W_d) = sum_c SwiGLU(X, W_g[c], W_u[c], W_d[:, c])`:
- **`SmolLM2-135M` (`K=576, H=1536, N=576`)**:
  - Decode (`M=1`): **`1 x (1, 576, 1536, 576)`** (**1 VII call/layer** instead of 3 + CPU SiLU/MUL!).
  - Prefill (`M=64`): **`3 x (64, 576, 512, 576)`** (exact split, `0` padding!).
- **`Qwen2.5-1.5B` (`K=1536, H=8960, N=1536`)**:
  - Decode (`M=1`): **`9 x (1, 1536, 1024, 1536)`** (last 256 channels zero-padded: `gate=0 -> SiLU(0)=0, up=0 -> prod=0`, exact `0` contribution, 9 VII calls instead of 15 + zero CPU SiLU/MUL!).
  - Prefill (`M=64`): **`18 x (64, 1536, 512, 1536)`** (last 256 channels zero-padded).
- **Probe-A Dynamic `param_slice` Swapping**: `ext[2]` (`[W_gate | W_up | W_down_eff]`) is content-hashed once and pinned in `weight_cache` (respecting Android LMKD `MemAvailable/3` budget); the 11-chunk graph `.dgc0` depends ONLY on `(shape, s_gate_grid, D_gate, D_up, D_mul, D_down)` and is shared across all 28/30 layers!

---

## 7. Task 4 Audit: DMA Bandwidth Ceiling (`~28 GB/s`) & `0x940`/`0x980`/`0xa40` Attributes

Offline audit of [`docs/isa-dma/dma_ttu_bitfields.md`](../isa-dma/dma_ttu_bitfields.md), `compiler__buenos_a0_config.txt`, and `extracted/vendor/lib64/libedgetpu_tflite_compiler.so`:

1. **No Hidden Burst / AXI Throughput Knob in `0xa40`, `0x940`, or `0x980`**:
   - Every active bit of `0xa40` (`HibDma.FlexibleDma`, verified `184/184` + `FUN_02ce69fc`), `0x940` (`RingBusProducer`, `110/110` + `18/18` Phase 7), and `0x980` (`RingBusConsumer`, `338/338` + `184/184` Phase 7) is accounted for. None carries a burst-length, outstanding-request, or AXI QoS bandwidth multiplier.
   - `config.scalar_core().parameter_memory_width_bits() == config.bus().ring_bus_width_bits() == 256` (`32 B/cycle`). At `1,286.4 MHz` (`PM=5`, `core=0x19`, `mem=0x15` + FW WakeLock), raw single-core ring peak is `41.16 GB/s`, and our observed **`27.6–28.0 GB/s`** sustained weight-streaming rate is the hardware SoC fabric / LPDDR5X single-core read ceiling.
2. **Only Two Architectural Paths Exceed `28 GB/s` Effective Weight Bandwidth on `buenos_a0`**:
   - **True 4-bit nibble-packed weights (`0x980 w8 = (1 << 21) | (zp << 22)`, `2x` = `~55 GB/s` equiv.)** via the `W4A16(BF16)` path (`CSR 0x56 = 0x26`, proven bit-exact in Phase 5).
   - **Hardware Huffman weight decompression (`platforms.darwinn.isa.HibDma.FlexibleDma.Huffman` / `buenos_a0_multimedia_huffman_config.binarypb`)** for constant DRAM parameters (`1.25x–1.5x` entropy compression on INT8 weights).

---

## 8. Live `rango` Silicon Verification of Standalone Synthesized Probes

Verified on `rango` via zero-Google-`.so` `tpu_baremetal_runner` + `bm_dvfs_probe` (`PM=5` + FW WakeLock, `--iters 20`, `MAPS_GOOGLE=0`; all probes use fresh random weights/inputs/biases and all-distinct non-PoT scales, `work/device-runner/phase7_perf/swiglu_probe_verify/summary.json`):

| Standalone Synthesized Probe | Shape (`M x K x H x N`) | `ext[2]` | `max_abs_err` | Exact Bytes | `PM=5` `hw_us` med (mean) | `wall_ms` | GOPS | Weight BW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `probe_swiglu_1x64x64x64_nonpot` | `1 x 64 x 64 x 64` | `13,056 B` | **`0`** | **`64 / 64`** | **`27.0 (27.9)`** | `0.282` | `0.9` | `0.46 GB/s` |
| **`probe_swiglu_1x576x1536x576_smollm2`** | **`1 x 576 x 1536 x 576`** | **`2,668,800 B`** | **`0`** | **`576 / 576`** | **`118.0 (119.4)`** | **`0.561`** | **`45.0`** | **`22.50 GB/s`** |
| **`probe_swiglu_1x1536x512x1536_qwen512`** | **`1 x 1536 x 512 x 1536`** | **`2,369,536 B`** | **`0`** | **`1536 / 1536`** | **`102.0 (111.0)`** | **`0.359`** | **`46.3`** | **`23.16 GB/s`** |
| **`probe_swiglu_1x1536x1024x1536_qwen1024`** | **`1 x 1536 x 1024 x 1536`** | **`4,732,928 B`** | **`0`** | **`1536 / 1536`** | **`180.0 (192.2)`** | **`0.476`** | **`52.4`** | **`26.23 GB/s`** |
| `probe_swiglu_64x576x512x576_smollm2_pp` | `64 x 576 x 512 x 576` | `910,336 B` | **`0`** | **`36,864 / 36,864`** | **`144.0 (145.9)`** | **`0.364`** | **`786.4`** | `6.24 GB/s` |
| **`probe_swiglu_64x1536x512x1536_qwen_pp`** | **`64 x 1536 x 512 x 1536`** | **`2,400,256 B`** | **`0`** | **`98,304 / 98,304`** | **`317.0 (318.9)`** | **`0.607`** | **`952.6`** | `7.53 GB/s` |

**Total standalone synthesized live silicon parity**: **`6 / 6` probes, `138,880 / 138,880` output bytes bit-exact (`max_abs_err = 0`), `MAPS_GOOGLE = 0`, zero intermediate DRAM spills.**
