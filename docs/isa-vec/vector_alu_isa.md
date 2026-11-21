# Vector, Quantization & Activation ISA (`laguna` / `buenos_a0`, Tensor G5)

Clean-room VLIW, CSR, spline-LUT, and live-hardware behavioral specification recovered from differential Edge TPU `.dgc0` binaries executed on `rango` across Baseline (`3`), Batch 0 (`9` vec/pipe2), Batch 1 (`19/33` unquantized), Batch 1b (`33/33` quantized), DMA shape sweeps (`1x1x1x1..512`), and `5/5` causal `--run-dgc0` patch proofs.

---

## 1. Instruction Stream Framing & VLIW Opcode Map

Chunks (`chunk[0]..chunk[last]`) contain a **64-bit little-endian qword stream** (instructions are multiples of `8 B`, **not** strictly 16 B aligned).

- **`[11:0]`**: 12-bit instruction opcode (`w0 & 0xfff`).
- **`[47:12]`**: Core / tile enable mask (`0x00001`..`0x0ffff` for 16 cores) on `0x3c0`, `0x440`, `0x5c0`, `0x4c0`, `0x640`, `0x400`, `0x200` (`[47:12]` carries sequence in `0x680`/`0x6c0`/`0x900`..`0xa40`).
- **Sequence counter**: `[63:48]` for `0x100`, `0x200`, `0x3c0`, `0x400(112B)`, `0x440`, `0x4c0`, `0x500`, `0x5c0`, `0x640`; `[43:28]` for `0x680`, `0x6c0`; `[27:12]` for `0x900`, `0x940`, `0x980`, `0x9c0`, `0xa40`.

| Opcode | Bytes | Qwords | Role (Observed) |
|---|---:|---:|---|
| `0x100` | `8` or var (`208..224`) | `1` or `26..28` | `1`-qword sync trailer (`w0 >> 24 == 0`), or **`MMA_CSR_WRITE`** (used by `MUL` Stage 1) |
| `0x180` | `8` | `1` | Sync / barrier trailer after `0x900` (`0x0000000000000180`) |
| `0x200` | `32..320` | `4..40` | **`VICA_WRITE_CSR`** (`32..224 B` CSR list), or `320 B` `HDR-DATA` tensor metadata/scaling packet (`w2 == 0`) |
| `0x3c0` | `224` | `28` | Tile / input setup header (`w25..26` carry `(zp_s8 << 8) \| zp_s8` padding pattern) |
| `0x3e0` | `8` | `1` | Final program halt (`0x00000000008003e0`, `finish_and_sleep`) |
| `0x400` | `112` or `8` | `14` or `1` | Output drain / store descriptor (`112 B`) in `chunk[last]`, or `8 B` loop trailer |
| `0x440` | `128` | `16` | Tile descriptor / spatial stride setup |
| `0x4c0` | `160` | `20` | Fixed VICA thread/compute loop descriptor (`160 B`; identical payload across `RELU`, `QUANTIZE`, `ADD`, `MUL`, `LOGISTIC`, `FC`) |
| `0x500` | `336..352` | `42..44` | Fused `MMA`/FC compute packet (`fc_act_*`); embeds `0x55..0x61` post-op CSRs in tail (`w38..w41`) |
| `0x5c0` | `208` | `26` | Fixed VICA SIMD engine descriptor (`208 B`; identical payload across `RELU`, `QUANTIZE`, `ADD`, `MUL`, `LOGISTIC`, `FC`) |
| `0x640` | `208` | `26` | **`LoadCoefficientTables`**: 47 IEEE-754 `float32` entries (`8 x 5` coeffs + `7` breakpoints) packed at bit `76` |
| `0x680` | `32` | `4` | Guard / sync descriptor before `0x900` (`[47:12]` carries mask `0x0ffff`) |
| `0x6c0` | `16` | `2` | Per-tile lane/bank step (`[1,64,1,64]` & `[1,16,16,16]` loops, `w1` carries core/offset stepping) |
| `0x900` | `72` | `9` | Event/barrier sync (`0x000fffff4000<m>900` in tail) paired with `0x100`/`0x180`/`0x3e0`/`0x400` |
| `0x940` | `48` | `6` | SRAM/TTU ring descriptor (handoff to next chunk) |
| `0x980` | `80` | `10` | Quarter-tile / bank descriptor |
| `0x9c0` | `64` | `8` | Output DMA tail setup (`chunk[last]`) |
| `0xa40` | `64` | `8` | Flexible DMA transfer (`in`, `par`, `out`) |

---

## 2. Two-Stage Vector Execution Model

Observed across every standalone unary/binary vector op (`QUANTIZE`, `RELU`, `ADD`, `MUL`, `LOGISTIC`):

1. **Vector Ops Dispatch Identical `0x5c0`/`0x4c0` Loops**:
   Comparing `[1,1,1,16]` blobs (`relu_1x1x1x16`, `quant_1x1x1x16`, `add_1x1x1x16`, `logistic_1x1x1x16`) shows `0x3c0`, `0x440`, `0x5c0`, and `0x4c0` are **byte-for-byte identical** across ops. Operation differentiation lives in:
   - `0x640` coefficient tables (`LoadCoefficientTables`)
   - `0x200` `HDR-DATA` (`320 B`) & `VICA_WRITE_CSR` register streams (`0x56..0x61`)
   - `0x100` `MMA_CSR_WRITE` (exclusively used by `MUL` Stage 1)
   - `0x500` embedded CSRs (fused `FC` + `NONE`/`RELU`/`RELU6`).
2. **Stage 1 vs Stage 2 Rescale & Clamp (`0x200`/`0x100`)**:
   - **Stage 1** (`@58` in `QUANTIZE`, `@98` in `ADD`, `@100` `0x100` in `MUL`): applies the quantized scale multiplier `M1` (`s_in / s_out` for `QUANTIZE`; normalized ratio for `ADD`; `(s0*s1)/s_out` for non-PoT `MUL`) and clamps in `zp_out`-shifted quantized space `[(-128 - zp_out), (+127 - zp_out)]` (or `[0, +96]`/`[0, +127]` for `RELU`/`RELU6`).
   - **Stage 2** (`@73` in `QUANTIZE`, `@114` in `ADD`, `@116` in `MUL`): rescales/dequantizes via `mul = s_out` (`0x5b..0x5c`) and clamps in real space `[(-128 - zp_out)*s_out, (+127 - zp_out)*s_out]`, paired with an output `0x640` LUT (`slope = 1 / s_out`, `bp0 = -zp_out * s_out`) and `0x57 = ((zp_out + 128) & 0xff) << 3`.
   - **Causal Runtime Proof (`p2_q_s2_mul_half_x2`)**: Patching **only** Stage 2 `0x200@73` `mul` from `0.0625` to `0.125` on `q_s_0_5` doubled every Stage 1 rounded integer output (`q_mid * 0.125 * 16.0`, `4096/4096` exact), proving both stages execute sequentially on hardware at runtime rather than Stage 2 being dead metadata.
3. **Hardware Arithmetic & Rounding Law (FP32 RNE)**:
   - **Rounding**: Hardware performs **IEEE-754 `float32` Round-Half-to-Even (`np.rint`)**, **not** TFLite's Round-Half-Up (`floor(x + 0.5)`), **not** integer fixed-point `(m, shift)`, and **not** `float64`/`bf16`.
   - **1-ULP Silicon Proof**: For `q_s_nonpot_1_5` (`s_in=0.045, s_out=0.03`), `np.float32(0.045)/np.float32(0.03) = 0x3fc00001` (`1.50000012`); `f64_rne` matches `3382/4096` while `f32_rne` matches **`4096/4096`** (`#46` `p1_q_s1_mul_05_to_15` also reproduced this `+1 ULP` tie-break `4096/4096` on `-51 -> -77`, `25 -> +38`).
   - **`ADD` Tie-Break Rule (`add_nonpot` `4096/4096`)**: Hardware rounds `(x0 - zp0) * f32(s0/s_out) + (x1 - zp1) * f32(s1/s_out)` to nearest even **before** adding integer `zp_out` (`4096/4096` vs `3200/4096` if `+ zp_out` precedes `rint`), shifting `.5` ties to odd outputs whenever `zp_out` is odd (`zp_out = -3`).

---

## 3. Exact `0x200` / `0x100` CSR Packing & `0x50..0x61` Bitfields

### 3.1 32-bit CSR Slot Format (25-bit Register Payload!)

Each nonzero 32-bit slot in a CSR-write `0x200` (`VICA_WRITE_CSR`) or `0x100` (`MMA_CSR_WRITE`) payload (and `w19..w41` of `0x500`) has **no parity/flag bit at bit 7**:

```text
 31                           7 6         0
+------------------------------+-----------+
|          val25 (25 bits)     | reg7 (7b) |
+------------------------------+-----------+
  val25 = (u32 >> 7) & 0x1ffffff
  val24 = (u32 >> 8) & 0xffffff   (val25 >> 1)
  b7    = (u32 >> 7) & 1          (val25 & 1 -> explains 0xdc..0xe1 slot bytes)
```

Header register **`reg 0x50`** (in `0x200`) / **`reg 0x70`** (in `0x100`) encodes the CSR batch length:
`val24 = (num_csr_writes << 5) | 0x10` (e.g., `4` writes -> `0x90`, `5` writes -> `0xb0`, `47` writes -> `0x5f0`, `51` writes -> `0x670`).

### 3.2 Continuous 125-bit Stream (`0x5b..0x5f`)

Concatenating the five 25-bit registers `0x5b..0x5f` little-endian (`stream125 = sum(val25(0x5b + i) << (25*i) for i in 0..4)`):

| Bit Slice in `stream125` | Width | Reg Span | Field / Meaning |
|---|---:|---|---|
| `[12:0]` | 13b | `0x5b[12:0]` | Stage flags: `0x0000` in Stage 1 (`0x5b=0x000000` mod `0x2000`), **`0x0080` in Stage 2** (`0x5b_val24[6] = 1`, `0x000040`) |
| **`[44:13]`** | **32b** | `0x5b[24:13]` + `0x5c[19:0]` | **IEEE-754 `float32` Multiplier (`mul_f32`)**: Stage 1 = `f32(M1)`, Stage 2 = `f32(s_out)` |
| `[45]` | 1b | `0x5c[20]` | Reserved / `0` across all observed sweeps |
| **`[77:46]`** | **32b** | `0x5c[24:21]` + `0x5d[24:0]` + `0x5e[2:0]` | **IEEE-754 `float32` Upper Clamp (`fclip_max`)**: Stage 1 = `f32(qmax - zp_out)` (e.g., `127.0` = `0x42fe0000`, `RELU6` `96.0` = `0x42c00000`); Stage 2 = `f32((qmax - zp_out)*s_out)` (e.g. `6.35` = `0x40cb3333`) |
| `[82:78]` | 5b | `0x5e[7:3]` | Observed `0x00` across all sweeps |
| **`[114:83]`** | **32b** | `0x5e[24:8]` + `0x5f[14:0]` | **IEEE-754 `float32` Lower Clamp (`fclip_min`)**: `0.0` (`0x00000000`) for `RELU`/`RELU6`; `f32(-128 - zp_out)` in Stage 1 (`NONE`); `f32((-128 - zp_out)*s_out)` in Stage 2 (`NONE`) |
| `[124:115]` | 10b | `0x5f[24:15]` | Observed `0x000` (except `0x300` in `add_nonpot @98`, i.e., `0x5f_val24[23:22] = 0b11`) |

> [!IMPORTANT]
> **CSR Omission / Reset Semantics (`0x5e` & `0x5c`)**:
> - In PoT Stage 1 (`0.5`, `0.625`, `2.0`, `ADD`), `0x5e = 0x000001` (`fclip_max[31:29] = 0b010`) is often **omitted**, yet live silicon (`q_s_2_0`, `4096/4096`) still upper-clamps at `+127` (`0x42fe0000`), whereas explicitly writing `0x5e = 0` wipes `fclip_max[31:29]` to `0` (`probe_0x5e_bit1_zero`).
> - In `MUL` with `s0=s1=s_out=0.0625` (`mul_base_0625`, `mul_act_relu/6`), `0x5c` is omitted from both `0x100@100` and `0x200@116` (`4096/4096`), whereas `mul_nonpot` emits `0x5c = 0x01ec44` (`0.066667f`) in `0x100@100` and `0x5c = 0x41e7ae` (`0.03f`) in `0x200@116`. When `0x5c` is present in `0x100@100`, patching it (`p3_mul_nonpot_s1_to_05`) directly sets the product scale (`4096/4096`).

### 3.3 Companion Vector CSRs (`0x56..0x61` & `0x00`/`0x05`/`0x17`)

| Reg | Observed Values (`val24`) | Bitfield / Role |
|---|---|---|
| `0x56` | `0x000608` (`ADD` S1), `0x000700` (S2), `0x080700`/`0x087700` (`MUL` `0x100`), `0x080000` (`FC`) | Stage / pipeline mode control (`ADD` S1 vs S2 vs `MUL`/`FC`) |
| **`0x57`** | `((zp_out + 128) & 0xff) << 3` | **Output `uint8` zero-point** (`zp_out=0` -> `0x400`; `-2` -> `0x3f0`; `-3` -> `0x3e8`; `-20` -> `0x360`; `0x000` in Stage 1 or signed `RELU`) |
| `0x58` | `0x028000`/`0x014000` (S1), `0x002000`/`0x004000` (S2), `0x012000` (`FC`) | Tile/accumulator width or stage select |
| `0x60` | `ADD` S1 only: `0xfffffc` (`-4`, sym), `0xfffffa` (`-6`, asym `0.5:1`), `0xfffff1` (`-15`, `add_nonpot`); `0` in S2 | Binary `ADD` Stage 1 offset/prescale parameter (exact algebraic derivation unresolved; see Unknowns) |
| **`0x61`** | `QUANTIZE`: `0x100 + 2*zp_out` (`0x1fe` for `+127`, `0x102` for `+1`, `0xd8` for `-20`); `ADD`: `0x101+2*zp_out` (S1), `0x100+2*zp_out` (S2); `FC`: `0x020100 + 2*zp_out` | **Stage 1 / `FC` signed output zero-point** (`val25[10:2] = (128 + zp_out)`, verified live by `p5_q_zp_out_1_to_10`). Bit 1 (`b7=1` in `0xe1`) toggles on `ADD` S1. Bit 17 (`0x020000`) sets in `FC` `0x500`. |
| `0x00/0x05/0x17` | Present in small `[1,1,1,16]`/`[1,1,1,64]` `0x200` and `MUL` `@116` | Loop/stride registers (`0x17`: `0x002000` for `N=16`, `0x020000` for `N=64`, `0x800000` for `1x16x16x16`) |

---

## 4. Activation Fusion: Standalone vs Fused (`ADD`, `MUL`, `FC`)

| Pattern | `act` | Blob Size | Exact ISA Diff vs `NONE` |
|---|---|---:|---|
| `add_base_0625` -> `add_act_relu` | `NONE -> RELU` | `668,288 B` (`+0 B`) | `c5 0x200@98` `0x5f`: `0x0030c0` (`-128.0f`) -> **`0x000000` (`0.0f`)** |
| `add_act_relu` -> `add_act_relu6` | `RELU -> RELU6` | `668,288 B` (`+0 B`) | `c5 0x200@98` `0x5d`: `0x17f000` (`+127.0f`) -> **`0x160000` (`+96.0f`)** |
| `mul_base_0625` -> `mul_act_relu` | `NONE -> RELU` | `667,968 B` (`+0 B`) | `c5 0x100@100` `0x5f`: `0x0030c0` -> **`0x000000`** |
| `mul_act_relu` -> `mul_act_relu6` | `RELU -> RELU6` | `667,968 B` (`+0 B`) | `c5 0x100@100` `0x5d`: `0x17f000` -> **`0x160000`** |
| `fc_act_none` -> `fc_act_relu` | `NONE -> RELU` | `656,064 B` (`+0 B`) | `c3 0x500` `w41.lo` (`0x5f`): `0x0030c05f` -> **`0x0000005f`** |
| `fc_act_relu` -> `fc_act_relu6` | `RELU -> RELU6` | `656,064 B` (`+0 B`) | `c3 0x500` `w40.hi` (`0x5d`): `0x17f0005d` -> **`0x1600005d`** |
| `pipe2_fc_relu_1x16x16` (Batch 0) | Separate 2-op | `656,512 B` (`+480 B`) | `c3 0x500` keeps `NONE`, `c5` grows `1456 -> 1936 B` for standalone `ReLU` pass |

---

## 5. `0x640` LUT & `LOGISTIC` Piecewise Minimax Spline

Every `0x640` (`208 B`, 26 qwords) packs **47 IEEE-754 `float32` entries** starting at bit `76` (bit `12` of qword `1`, stride `32` bits):
- Entries `0..39`: `8` segments x `5` polynomial coefficients `[c0, c1, c2, c3, c4]` evaluated via degree-4 Horner `c0 + x*(c1 + x*(c2 + x*(c3 + x*c4)))`.
- Entries `40..46`: `7` strictly ascending breakpoints separating the `8` segments (`x < bp[i]`).

### 5.1 Two-Region Affine Sign-Split LUT (`QUANTIZE`, `RELU`, `ADD`, `MUL`, `LOGISTIC` in/out)

Affine stages emit a 2-region `0x640` table (`bp[1..6] = +inf`) that converts signed `int8` <-> `uint8` (`+128/-128` offset) at slope `1/s`:
- **`bp[0] = -zp * s`** (e.g., `0.0` for `zp=0`; `+0.3125` for `zp=-10, s=0.03125`; `+0.15` / `-0.28` / `+0.06` in `mul_nonpot`; `-0.0625` in `q_zp_in0_out1`; `+4.0` in `relu_1x1x1x16` where output `uint8 zp = -128`).
- `seg0` (`x < bp[0]`): `[+128.0, 1/s, 0, 0, 0]`; `seg1..7` (`x >= bp[0]`): `[-128.0, 1/s, 0, 0, 0]`.
- Verified causally by **`p5_q_zp_out_1_to_10`**: leaving `0x640@51` unpatched at `bp0 = -0.0625` while patching `0x61` to `zp_out=10` caused elements in `[-0.625, -0.0625)` (`-10 <= x_in <= -2`) to saturate at `-1` (`seg0` negative branch cap), while `x_in >= -1` (`x_real >= bp0`) and `x_in <= -11` shifted by `+9` (`-1 -> 9`, `0 -> 10`, `-15 -> -5`).

### 5.2 `LOGISTIC` Pipeline & Sigmoid Coefficients

`LOGISTIC` executes **three** `(0x5c0, 0x4c0, 0x640, 0x200)` stages:
1. **Input Dequant Stage**: `0x640` linear `[+/-128, 1/s_in]` with `bp0 = -zp_in * s_in`, followed by `0x200` `mul = s_in`, `fclip_max = +11.1` (`0x4131999a`), `fclip_min = -11.1` (`0xc131999a`).
2. **Sigmoid Spline Stage (`0x640`)**: Identical across all 4 `LOGISTIC` models (`logistic_1x1x1x16`, `logistic_scale_zp`, `vec_logistic_1x16x16x16`, `pipe2_add_logistic_1x16x16x16`), approximating `256 * sigmoid(x)`:
   - **Breakpoints**: `[-6.05023, -3.37245, -1.54011, 0.13833, 1.92349, 4.96436, 8.72828]`
   - `seg0`: `[24.215595, 9.885680, 1.523298, 0.104707, 0.002703]`
   - `seg1`: `[120.863708, 74.884506, 18.168011, 2.025888, 0.086899]`
   - `seg2`: `[139.329712, 89.448639, 21.420982, 2.024681, 0.037479]`
   - `seg3`: `[128.003616, 64.002953, -0.260223, -6.271765, -1.161241]`
   - `seg4`: `[127.991402, 64.018196, 0.264764, -6.304061, 1.177963]`
   - `seg5`: `[110.976654, 98.575623, -26.815042, 3.418226, -0.170363]`
   - `seg6`: `[200.260223, 27.220001, -5.073759, 0.425792, -0.013523]`
   - `seg7`: `[255.826904, 0.0, 0.002377, 0.0, -0.000008]`
3. **Output Requant Stage**: `0x640` `[+/-128, 256.0]` with `bp0 = 0.5`, followed by `0x200` `mul = 1/256` (`0.00390625`, `0x3b800000`), `fclip_max = 255/256` (`0.99609375`, `0x3f7f0000`), `fclip_min = 0.0`.
- **Host FP32 Spline Verification**: Evaluating this exact spline in FP32 (`np.rint(y) - 128`) matches live hardware **16/16** (`logistic_1x1x1x16`) and **4096/4096** (`logistic_scale_zp`).

---

## 6. Shape Tiling, Core Masks & `finish_and_sleep` Tail

Observed across `[1,1,1,1..512]`, `[1,64,1,64]`, and `[1,16,16,16]`:
- **32 Elements per Core Slot (16 Cores Total = 512 Elements/Pass)**:
  In `chunk[last]`, each `0x400(112B)` + `0x400(8B)` output store covers up to `32` elements (`w6`/`w7` encode byte length - 1 or end-lane count: `0x0400` for `1`, `0x1e00` for `16`, `0x3e00` for `32`):
  - `N = 1..32`: `1 x 0x400(112B)`, core mask `0x00001`
  - `N = 48, 49, 64`: `2 x 0x400(112B)`, core masks `0x00001, 0x00002`
  - `N = 65`: `3 x 0x400(112B)`, masks `0x00001, 0x00002, 0x00004` (`0x3c0` x `3`, `0x00007` on `0x440`)
  - `N = 128`: `4 x 0x400(112B)`, masks `0x00001..0x00008`
  - `N = 256`: `8 x 0x400(112B)`, masks `0x00001..0x00080`
  - `N = 512`: `16 x 0x400(112B)`, masks `0x00001..0x08000` (all 16 cores)
- **2D / 4D Tiling (`[1,64,1,64]` vs `[1,16,16,16]`)**:
  - `[1,64,1,64]` (`4096` B): Uses 4-core stride mask **`0x01111`** (cores `0,4,8,12`) on `0x440`/`0x6c0`, outputting `64` passes x `0x400(112B)` (`chunk[last]` = `6112 B`).
  - `[1,16,16,16]` (`4096` B): Uses all-core mask **`0x0ffff`** across `16 x 0x400(112B)` + `64 x 0x400(8B)` (`chunk[last]` = `7456 B`).
- **Padding Zero-Point in `0x3c0` (`w25..w26`)**:
  For odd or padded tensors, `0x3c0` `(w26[11:0] << 4) | w25[63:60]` holds the replicated 16-bit padding byte `(zp_u8 << 8) | zp_u8` (e.g. `w25[63:60]=0x6, w26[11:0]=0xf6f -> 0xf6f6` for `zp=-10` in `logistic_scale_zp` & `b2_q_1x1x1x1`, `0xfbfb` for `zp=-5` in `pipe2_fc_relu_1x16x16`; `0` for `zp=0`).
- **`finish_and_sleep` Tail (`chunk[last]` last 8 instructions)**:
  Every compiled TPU blob ends identically with:
  1. `2 x (0x680@seq, 0x900@(seq+1), 0x180)` (`0x680` `w1` = `0x000ffff8000003c2`, then `0x000ffff8000003c0`; `0x900` `w1 = 0xff80000040000000`, `w2 = 0x7f`)
  2. `0x900` (`w0 = 0x000fffff4000<m>900`, `w1 = 0xff80000000000000`, `w2 = 0x7ff`; `m` varies `0..9` per graph/shape)
  3. `0x3e0` (`w0 = 0x00000000008003e0`).

---

## 7. Observed vs Inferred vs Unknowns Ledger

| Status | Item | Evidence / Basis |
|---|---|---|
| **Observed** | 25-bit CSR slot layout `(val25 << 7) \| reg7` & `0x50/0x70 = (count<<5)\|0x10` | Byte-exact match across all Batch 1/1b `0x200`/`0x100`/`0x500` packets |
| **Observed** | `stream125`: `mul_f32@[44:13]`, `fclip_max@[77:46]`, `fclip_min@[114:83]` | `delta=0` across all `QUANTIZE`/`ADD`/`MUL`/`LOGISTIC`/`FC` + causal probes `p1..p4` (`4096/4096` each) |
| **Observed** | Hardware rounding = IEEE-754 FP32 RNE (`np.rint`) + post-rounding `+ zp_out` | `q_s_nonpot_1_5` (`0x3fc00001`), `p1` (`4096/4096`), `add_nonpot` (`4096/4096`), all PoT/non-PoT `QUANTIZE`/`ADD`/`MUL` |
| **Observed** | Stage 1 + Stage 2 dual execution (`mul_f32` -> `s_out` -> `0x640 1/s_out`) | `p2_q_s2_mul_half_x2` (`4096/4096` match when patching Stage 2 `0x5c`) |
| **Observed** | `FC` activation fusion in `0x500` (`w40` `0x5d`, `w41` `0x5f`) at `+0 B` | `fc_act_{none,relu,relu6}` (`656,064 B`) vs `pipe2_fc_relu` (`656,512 B`) |
| **Observed** | `0x640` 47-float table (`bp0 = -zp*s`, `LOGISTIC` 8-seg deg-4 sigmoid) | `16/16` (`logistic_1x1x1x16`), `4096/4096` (`logistic_scale_zp`), `p5` negative `seg0` saturation |
| **Observed** | `32` lanes/core output chunking across `16` cores & `chunk[last]` 8-insn tail | `b1_q_1x1x1x{4..512}`, `b2_q_1x1x1x{1..65}`, all Batch 0/1/1b `.dgc0`s |
| **Inferred** | VICA resets `0x5e[0] = 1` (or preserves prior `0x5e`) when `0x5e` is omitted | `q_s_2_0` omits `0x5e` yet clamps `+127` (`4096/4096`); writing `0x5e=0` breaks `fclip_max` (`#41`, `#45`) |
| **Inferred** | PoT `MUL` (`M=0.0625` / `1.0`) derives scaling via `0x640` (`1/s`) + `HDR-DATA` when `0x5c` is omitted | `b1:mul_act_relu` (`1.0`) and `b1b:mul_act_relu` (`0.0625`) omit `0x5c` in `0x100`/`0x200` yet produce exact scaled products (`4096/4096`) |
| **Unknown** | Exact subfields of `0x56`, `0x58`, `0x00`/`0x05`/`0x17`, and `0x60` (`-4/-6/-15` in `ADD` S1) | Observed correlation with stage/stride/asymmetric ratios; no single-bit ablation yet |
| **Unknown** | Exact bit packing inside `HDR-DATA` words `w3..w35` (besides `s_in`/`zp_in`/`min`/`max` in `w35..39`) | Multi-op tensor descriptors shared across DMA/VICA |
| **Unknown** | Event index `m` (`0..9`) semantics in tail `0x900` (`0x000fffff4000<m>900`) | Observed `m` varying with chunk/DMA count (`0`..`9`) across shapes |
