# Tensor G5 (`buenos_a0`) INT4 (`W4A8`) Packing, Per-Channel Scales & Targeted Kernel Fusion Spec

Clean-room ISA specification, verification proof, and C++17 emitter design for Owner Execution Order Steps 3 & 4 (`libopenbuenos` / `ggml-edgetpu`).

---

## 1. End-to-End Offline Verification Summary

| Suite / Artifact | Scope | Result | SHA-256 (prefix) |
|---|---|---|---|
| [`pack_int4.py`](../../tools/pack_int4.py) | Unpacked W4A8, 32B/64B per-channel scale+bias tail, true 4-bit nibble packing, 7-chunk W4A8 VLIW synth, CSR `0x56` probes | **PASS** (round-trip exact; live `buenos_a0`: `0x0010000` **PASS** `max_abs_err=1`, `0x0010070` nibpack saturated `[127]*16`) | `78111de21db4` |
| [`verify_int4_19.py`](../../tools/verify_int4_19.py) | All **19** `phase5_sweep_int4` models (`18` compiled + `1` compile-rejected W4A16) | **19/19 PASS**: `18/18` `ext[2]` bit-exact, `11/11` 7-chunk `c0..c6` bit-exact, `6/6` 9-chunk W4A8 exact, `1/1` `dequant_i4_fc` exact, `1/1` W4A16 reject | `ab870126ab7e` |
| [`verify_fusion_lut.py`](../../tools/verify_fusion_lut.py) | Live `buenos_a0` spline parity (`silu/gelu/exp`), Mode-A zero-overhead `GEMM + SiLU/GELU` probes (`4` shapes), multi-`0xa40` & `pipe2_*` chaining | **PASS + LIVE SILICON VERIFIED**: `12,288/12,288` spline bytes exact + **4/4 fused GEMM+SiLU/GELU probes 100% bit-exact (`max_abs_err=0`)** via zero-Google-`.so` `tpu_baremetal_runner` | `1f7b00fe1f7c` |
| `emit_dequant_i4_probe.py` | Standalone `dequant_i4_fc` (`1x64x32`, `CSR 0x56=0x0000026`, `0x980 w8=0x02200000`) & `uint8` `W8A16 (BF16)` unlock probes | **PASS** (full `DGC0`/`DIVE`/`ext[2]`/VLIW verification; `32/32` bit-exact FP32 `max_abs_err=0.0` once expanded to `2048 B u8`) | `f55a4f866e9b` |

---

## 2. Step 3 — INT4 (`W4A8`) Parameter Packing, Per-Channel Scales & `CSR 0x56`

### 2.1 Three Observed `ext[2]` Layouts (`18/18` Bit-Exact)

1. **W4A8 Unpacked Tile Weights (`16/16` compiled `fc_i4_pt_*`, `fc_i4_pc_*`, `fc_i4_blk*`)**:
   - **Critical Finding `[OBSERVED]`**: For TFLite `TensorType_INT4` (`W4A8`) `FULLY_CONNECTED`, the stock compiler does **NOT** nibble-pack `ext[2]`. It unpacks every signed weight `w in [-8, +7]` into a **full 8-bit byte `u = (w + 8) & 0xFF in [0x00..0x0F]`** in the **exact same tile layout** as INT8 (`[n_groups, pad4(K)//4, n_blk, 4]`, `n_blk = 32 if N > 32 else max(8, ceil8(N))`, `n_groups = ceil(N/32)`, active pad `0x08`, unused channel pad `0x00`).
   - **Bandwidth savings for compiler W4A8 FC = `0.00%`** (`ext[2]` weight bytes == INT8 `ext[2]` weight bytes).
   - **BlockwiseQuantization (`details_type=2`, `fc_i4_blk32/64/128`) `[OBSERVED]`**: The compiler silently ignores `BlockwiseQuantization`, drops the FP16 `[N, K//bs]` scale tensor, and emits a plain 7-chunk per-tensor W4A8 kernel with **`s_w = 1.0`** (`11/11` bit-exact verified).

2. **Separate-DMA Per-Channel FP32 Scales & Bias Tail (`6/6` 9-chunk W4A8 models)**:
   - Triggered whenever `per_channel` scales are present (`fc_i4_pc_*`) **or** non-zero bias is paired with large `K > 64` (`fc_i4_pt_1x128x32_bias`, `fc_i4_pt_1x256x64_bias`).
   - Appended directly after `weights_bytes` in `ext[2]` and streamed via a second `0xa40` DMA (`chunk[6]`), in `pad4(N) // 4` channel groups of 4 channels (`c = 4j .. 4j+3`):
     - **Per-channel scales only** (`0x61[19]=1, 0x61[18]=0`, `8 * pad4(N)` B): 32 B per 4 channels: `[4x LE fp32(s_in * s_w[c] / s_out), 16B zero]`.
     - **Separate-DMA bias only** (`0x61[19]=0, 0x61[18]=1`, `8 * pad4(N)` B): 32 B per 4 channels: `[4x LE int32(bias[c]), 16B zero]`.
     - **Interleaved scales + bias** (`0x61[19]=1, 0x61[18]=1`, `16 * pad4(N)` B): 64 B per 4 channels: `[4x LE fp32 scale, 16B zero, 4x LE int32 bias, 16B zero]`.

3. **True 4-Bit Nibble-Packed Weights + Raw FP32 Scales (`dequant_i4_fc_1x64x32`) `[OBSERVED]`**:
   - **Only** `DEQUANTIZE(INT4) -> FULLY_CONNECTED` emits true 4-bit nibble packing (`1024 B` weights + `128 B` raw FP32 `s_w[0..N-1]` = `1152 B` vs `2304 B` INT8+pc -> **50.00% bandwidth savings**):
     ```c++
     // Exact nibble layout (verified 1024/1024 B bit-exact on dequant_i4_fc_1x64x32):
     for (int kg = 0; kg < K / 2; ++kg) {
       for (int n = 0; n < N; ++n) {
         uint8_t lo = (uint8_t)((W[n * K + 2 * kg + 0] + 8) & 0x0F);  // even k -> bits [3:0]
         uint8_t hi = (uint8_t)((W[n * K + 2 * kg + 1] + 8) & 0x0F);  // odd  k -> bits [7:4]
         ext2_weights[kg * N + n] = (uint8_t)(lo | (hi << 4));
       }
     }
     // Followed by N contiguous LE float32 raw weight scales s_w[0..N-1] (4*N bytes).
     ```

### 2.2 `ext[2]` Bandwidth Savings Quantification

| Shape | INT8 `ext[2]` | Compiler W4A8 Unpacked | Savings | True 4-Bit Nibble Packed | Savings |
|---|---:|---:|---:|---:|---:|
| `1x16x16` per-tensor | `256 B` | `256 B` | `0.00%` | `128 B` | **`50.00%`** |
| `1x64x32` per-channel (`dequant_i4_fc`) | `2,304 B` | `2,304 B` | `0.00%` | `1,152 B` | **`50.00%`** |
| `1x128x64` per-channel + bias | `9,216 B` | `9,216 B` | `0.00%` | `4,608 B` | **`50.00%`** |
| SmolLM2-135M `q/o_proj` `576x576` pc | `336,384 B` | `336,384 B` | `0.00%` | `168,192 B` | **`50.00%`** |
| SmolLM2-135M `gate/up` `576x1536` pc | `897,024 B` | `897,024 B` | `0.00%` | `448,512 B` | **`50.00%`** |
| Llama-1B/3B `2048x2048` pc | `4,210,688 B` | `4,210,688 B` | `0.00%` | `2,105,344 B` | **`50.00%`** |
| Llama-1B/3B FFN `2048x5632` pc | `11,579,392 B` | `11,579,392 B` | `0.00%` | `5,789,696 B` | **`50.00%`** |

### 2.3 VLIW Bitfields: `CSR 0x56`, 9-Chunk `0x240` Per-Channel Pipeline & Silicon Saturation Finding

#### `CSR 0x56` (25-bit Register in `CSR_040` / `HDR_DATA`)

| Bits | Width | Field | Verified Values |
|---|---:|---|---|
| `[20:13]` | 8 b | `rhs_zero_point` (`zp_w`) | `0x80` (`128`, INT8), `0x08` (`8`, INT4), `0x00` (`0`, BF16/FP32) |
| `[12:9]` | 4 b | VICA stage/op mode | `0x0` (MAC), `0x7` (`0xe00`, VICA S2/S3), `0x6` (`0xc10`, ADD S1), `0x2` (`0x436`, FP32 rescale) |
| `[7]` | 1 b | Conv3x3 flag | `1` (`0x100080` in `conv3x3`), `0` (FC / Conv1x1) |
| `[6:4]` | 3 b | `rhs_data_type` | `0` (`FIXED_POINT8`), `2` (`BFLOAT` MAC), `3` (`BFLOAT/HALF` VICA), **`7` (`FIXED_POINT4`)** |
| `[3:0]` | 4 b | `lhs_data_type` | `0` (`FIXED_POINT8`), `6` (`SINGLE` FP32 in `dequant_i4_fc`) |

- **Critical Silicon Finding (`fc_i4_*` Saturation on `buenos_a0`, Live-Verified) `[OBSERVED]`**:
  - Full-file byte diff of `fc_i8_pt_1x16x16_id_ref.dgc0` vs `fc_i4_pt_1x16x16_id.dgc0` (655,808 B) proves the **only** functional differences are:
    1. `chunk[3] w285`: `CSR 0x56 = 0x0010070` (`zp_w=8, rhs_data_type=7`) vs `0x0100000` (`zp_w=128, rhs_data_type=0`).
    2. `ext[2]`: 256 B unpacked `w + 8` vs `w + 128`.
  - On live `buenos_a0` silicon:
    - **`probe_i4_1x16x16_csr56_0010000.dgc0` (`0x56 = 0x0010000`, `zp_w=8`, `rhs_data_type=0`) -> PASS**: `15/16` exact, `max_abs_err = 1` (matches INT8 ref), proving `0x56[20:13] = 8` works cleanly in `FIXED_POINT8` mode.
    - **`probe_i4_1x16x16_csr56_0010070_nibpack.dgc0` (`0x56 = 0x0010070`, `rhs_data_type=7`, nibble-packed `ext[2]`) -> SATURATED `[127]*16`**, proving `CSR 0x56[6:4] = 7` cannot decode nibble-packed INT4 on `buenos_a0`.
  - **Actionable Recommendation for `libopenbuenos`**:
    - Do **NOT** emit `CSR 0x56 = 0x0010070` on `buenos_a0`.
    - For immediate bit-exact W4A8 on `buenos_a0`, store `w_i4` (`[-8..7]`) as `w_i4 + 128` with `CSR 0x56 = 0x0100000` or `w_i4 + 8` with `CSR 0x56 = 0x0010000` (both live-proven on `buenos_a0` silicon; see [`../hardware-errata.md`](../hardware-errata.md)).

#### 2.4 `DEQUANTIZE(INT4) -> FC` Compiler Bug & `0x980` `1:2 UINT8 -> BF16` Pipeline (`CSR 0x56 = 0x0000026`) `[CAUSALLY_PROVEN]`

`dequant_i4_fc_1x64x32.dgc0` (`657,280 B`, 9 chunks `[320, 80, 64, 2192, 64, 1392, 64, 224, 736]` B) is the **only** compiler-emitted `.dgc0` in the 19-model sweep targeting FP32-I/O `BF16` MAC (`CSR 0x56 = 0x0000026`). Full packet decode and causal hardware ablation expose both why the stock model deadlocks ([`google-ai-edge/LiteRT#12178`](https://github.com/google-ai-edge/LiteRT/issues/12178)) and how native `W8A16 (BF16)` works on silicon:

1. **Container & DIVE IR Integrity (Identical to Working 9-Chunk Models)**:
   - `ext[1]` RV64 interpreter (`638,977 B`): **SHA-256 identical** to `fc_i4_pc_1x64x32_bias.dgc0`.
   - `DIVE` IR: exact same **28 ops**, including Op 18 `Add(parameter_0, +0x400)` (`+1024 B` offset in stock output) and all 4 `PutBits` (`ch=2, 4, 6, 8`, `byte_off=37`, `bw=64`) properly zeroed.
   - `DWS0` IO & `MultiExecutable`: `input [1,64]` `data_type = 7` (`256 B` -> **`FLOAT32`**, note `dgc0_dump.py` mislabels enum `7` as `FLOAT16`), `output [1,32]` `data_type = 7` (`128 B`, `FLOAT32`), `dive_scratch_192512`, `compiler_options_pb` byte-for-byte identical, `Executable.scalars` identical except `field_14 = 6464` cycles.
   - All 9 chunks packetize cleanly with monotonic sequence IDs `0x00..0x3d`.
2. **Exact `0x980` & `W8A16 (BF16)` VLIW Wiring**:
   - **Input FP32 -> BF16 conversion (`c[3]`)**: `f00: 0x240` (`CSR 0x56 = 0x0000036`: `lhs=6 (SINGLE), rhs=3 (BFLOAT)`, `scale = 1.0f`) converts `64x FP32` activations into `BF16` in SRAM, followed by `f113: 0x440` (`f00.hi=0x02c0000200000000`, `f02.lo=0x0110000700006220`) and `f121: 0x040` (**`CSR 0x56 = 0x0000026`**: `zp=0, rhs=2 (BFLOAT), lhs=6 (SINGLE/BF16)`).
   - **Causal Proof of `0x980 w8 = (1 << 21) | (zp << 22)` (`c[5] f03: 0x980`)**:
     - Across all `19` `phase5_sweep_int4` models and all `74` Phase-3 models, `dequant_i4_fc_1x64x32.dgc0` `c[5] f03: 0x980` is the **only** `0x980` descriptor with non-zero bits in `w8[31:8]`:
       ```text
       c[5] f07.lo (0x980 w8, global bits [575:512]) = 0x0000000002200000
         bit 21       : 1    (enable 1:2 UINT8 -> BFLOAT16 DMA weight expansion)
         bits [29:22] : 8    (unsigned 8-bit zero_point subtracted before BF16 cast: BF16(float(u8 - zp)))
       ```
     - Live silicon ablation (`1x16x16`, `P = 128 B`, `hw_us = 398`) proves **`w8` implements `1:2` `UINT8 -> BFLOAT16` (`1 B -> 2 B`), NOT 4-bit nibble unpacking (`16/16` exact `max_abs_err = 0`)**.
   - **Half-Width VICA Weight Reshape (`c[5] f24: 0x5c0` + `f37: 0x4c0`)**:
     - `0x5c0`: `f03.lo[47:40] = 0x06`, half beat count `f03.hi = 0x001fffe4_00007000` (`8` vs `16`), `f11.hi = 0x0100...`, `f12.lo = 0x1e0`.
     - `0x4c0`: `f00.hi = 0x00000002_00180000`, `f04.lo = 0x0080_007c_0007c000`, `f06.lo = 0x00002453_1f3e0000`, `f09.lo = 0x00070204_00000000`.
   - **Weight Load + BF16 MAC + Per-Channel FP32 Scale (`c[5] f47..f86`, `c[6..8]`)**:
     - `f47: 0x440` (`f00.hi=0x00c0000000000000`, `f02.lo=0x0800` = `4,096 B` / `2,048 BF16` weights) feeds `f55: 0x500` (`BF16` MAC), followed by `f64: 0x200` (`CSR 0x56 = 0x0000436`, `0x61 = 0x0080002` per-channel FP32 scale multiply using `128 B` scales DMA'd in `c[6]`/`c[7]`).
3. **Why Stock Compilation Stalls & How Native `W8A16 (BF16)` Succeeds**:
   - The stock compiler emitted **`P = 1,024 B`** of packed 4-bit nibbles into a **`1:2` `UINT8 -> BF16`** `0x980` expander (`1,024 B -> 2,048 B` instead of the `4,096 B` expected by `64x32` BF16 MAC `0x500`), hanging execution forever (`Wait rc=-1`).
   - Expanding `ext[2]` to **`P = 2,048 B` `uint8` (`w + 8`, `[K/2, N, 2]`) + `128 B` raw `FP32` scales at `+0x800`** delivers the full `4,096 B` `BF16` matrix and executes cleanly on silicon in **`411–426 us` (`32/32` bit-exact `FP32`, `max_abs_err = 0.0`)**. See [`../hardware-errata.md`](../hardware-errata.md) for the full analysis and `Q4_0` caching implications.

#### 9-Chunk Per-Channel Scale & Separate-Bias Pipeline (`c[0..8]`)

| Chunk | Role | Verified Invariants (`6/6` W4A8 9-chunk models) |
|---|---|---|
| `c[0..2]` | Prolog + Input DMA | Identical to Phase 4 (`c[1]` uses `m_eff = M/4` when `M>=4 && K>=128` 4-row striping); `c[2]` `0xa40` `f0_hi_word = 0x01` |
| `c[3]` | Input VICA + MAC `CSR_040` | `CSR_040` enters **unquantized INT32 accumulator mode**: `0x55=0x81800` (or `0x785800` 4-row), `0x5b=0x800`, `0x5c=0x3f800` (`1.0f`), `0x5d=0x1f80000`, `0x5e=0x3` (`+inf`), `0x5f=0x7fc0` (`-inf`), `0x61=0x0` |
| `c[4]` | Weight DMA (`output::parameter`) | `0xa40` `f0_hi_word = 0x22`, `nbytes = w_bytes` (`ext[2][0 : w_bytes]`) |
| `c[5]` | Weight load + MAC + **`0x240` Rescale** | `0x940 -> 0x980 -> 0x440 -> 0x500 -> 0x240(20f)` (`CSR 0x61 = (has_pc<<19) | (has_bias<<18) | ((zp_out+128)<<2)`, `0x5c = 0x3f800` if `has_pc` else scalar `M_eff`) + `0x940` (`f0.lo[47:32] = w_bytes >> 4`, moved to `c[7][0]` when `N>32 && M==1`) |
| `c[6]` | Scale/Bias DMA (`output.1::parameter`) | `0xa40` `f0_hi_word = 0x42`, `nbytes = tail_bytes` (DIVE `PutBits` patches base `parameter_0` IOVA; offset comes from `0x940`/`0x980`!) |
| `c[7]` | Receive scales + Stage-3 MSB-flip | `0x980` (`f0.lo[47:32] = w_bytes >> 3`) + `0x5c0 -> 0x4c0 -> 0x640 -> HDR_DATA(0x200)` (`0x56=0xe00`, `0x61=(zp_out+128)<<2`) |
| `c[8]` | Output DMA + Finish | `0xa40` `f0_hi_word = 0x63`, `nbytes = M * pad4(N)` + `0x9c0 -> 0x400 -> 0x900...` |

#### Phase 4 Edge Fix Found on `(2 <= M < 48, K == 64, N <= 32)`

In `fc_i4_pt_4x64x32` and `fc_i4_pt_8x64x32`, the `post3` `0x3c0` packet uses 2D rewind scaled by `(M - 1)`:
- `flit[3].hi = ((0x20000 - 4 * (M - 1)) << 32) | ((M - 1) << 8)`
- `flit[4].lo = 0x0000C00000000000ULL | ((0x04000000 - (M - 1) * 0x800) & 0x03FFFFFF)`

---

## 3. Step 4 — Targeted Kernel Fusion (`GEMM + 0x640 SiLU/GELU` & Chained Projections)

### 3.1 Live Silicon `0x640` Spline Parity (`12,288 / 12,288` Exact)

All 3 clean-room Remez degree-4 probes (`silu_probe`, `gelu_probe`, `exp_probe`) executed on live `rango` via both `tpu_baremetal_runner` (`maps_has_google_tpu_libs == false`, `461..600 us`) and `tpu_direct_runner` (`217..244 us`) with **`max_abs_err = 0`** across all **12,288 / 12,288** bytes, matching our FP32-RNE Horner model (`eval_0x640_horner_fp32`) **100% bit-identically**.

### 3.2 Fuse Point: In-Place `chunk[5]` `0x640` + `HDR_DATA` (`+0 B`, `0` Extra Cycles!)

Every Phase 4 GEMM **already executes a full VICA `0x640` + `HDR_DATA` pass in `chunk[5]`** (post-matmul accumulation, pre-DRAM store) solely to perform a degree-1 MSB-flip (`uint8 -> int8`):
1. Systolic MAC (`CSR_040`) writes `u_mid = clamp(RNE(acc * (s_in * s_w / s_mid)) + (zp_mid + 128), 0, 255)` into Tile NM SRAM (`act = NONE`).
2. `chunk[5]` `HDR_DATA` (`0x200`, final 48 B at `+272..+320`) dequantizes `u_mid` to exact FP32:
   - `x = clamp((u_mid - zp_in_hw) * s_mid, fclip_min, fclip_max)`
   - `b17 + 80` (`8b`, `CSR 0x57[11:4]`): `zp_in_hw = (zp_mid + 128) & 0xFF`
   - `b17 + 189` (`32b`, `stream125[44:13]`): `fp32(s_mid)`
   - `b17 + 222` (`32b`, `stream125[77:46]`): `fp32(fclip_max)`
   - `b17 + 259` (`32b`, `stream125[114:83]`): `fp32(fclip_min)`
   - `b17 + 328` (`8b`, `CSR 0x61[9:2]`): `zp_out_hw` (added after `0x640(x)` before `clamp(RNE(y), 0, 255)`)
3. **Mode A — Zero-Overhead 1-Stage Fusion (Recommended for `SiLU` & `GELU`)**:
   - Replace `chunk[5]`'s `0x640` in place (preserving `header_76b` `pid` & tile mask!):
     - Set `tail_84b = 0x7804` (degree-4 Horner enable, bits `[1663:1580]`).
     - Set `HDR_DATA` `b17 + 328 = 0` (`zp_out_hw = 0`).
     - Set `fclip_max = min(x_mid_hi, f^{-1}((127.0 - zp_out) * s_out))` — **hardware saturation guard** preventing positive `int8` overflow (`127 -> 128 = -128`).
     - Place exact `0x640` breakpoint(s) at `f(x)/s_out + zp_out == -0.5`:
       - On segments with `q_out < 0`: `c0' = c0 / s_out + zp_out + 256.0` (outputs `[128..255] = q_out & 0xFF`).
       - On segments with `q_out >= 0`: `c0' = c0 / s_out + zp_out` (outputs `[0..127] = q_out & 0xFF`).
       - All segments: `c_k' = c_k / s_out` for `k = 1..4`.
   - **Verified on 4 shapes (`1x16x16` SiLU/GELU `zp=0`, `1x16x16` SiLU `zp=-112`, `4x32x32` multi-tile SiLU `zp=0`)**: `+0 B` `chunk[5]`, `max_lsb_err <= 0.0283 LSB` across all 256 `u_mid` levels, **`max_abs_diff = 0`** vs exact FP64 ideal!
4. **Mode B — 2-Stage VICA inside `chunk[5]` (`+480 B`, verified on `pipe2_fc_relu_1x16x16.dgc0`)**:
   - Needed only when `f(x)/s_out + zp_out` saturates below `-128` or has >2 zero crossings:
     - Stage 1 (`0x5c0 -> 0x4c0 -> 0x680 -> 0x640[deg-4] -> HDR_DATA`): outputs continuous `u_act = clamp(RNE(f(x)/s_out + zp_out + 128), 0, 255)`.
     - Stage 2 (`0x680 -> 0x680 -> 0x5c0 -> 0x4c0 -> 0x680 -> 0x640[deg-1 MSB-flip] -> VICA_CSR`): flips MSB (`u_act ^ 0x80`) into final signed `int8`.

### 3.3 Chained Consecutive Projections (`GEMM1 -> GEMM2` & SwiGLU) Inside a Single `.dgc0`

Verified across `fc_i4_pc_*` (`6` models), `dequant_i4_fc_1x64x32`, and `pipe2_conv1x1_add_1x16x16x16`:

1. **Universal `0xa40` Multi-DMA Tag Formula**:
   ```c++
   // f0.lo[47:32] in every 0xa40 host/parameter DMA packet:
   uint16_t f0_hi_word = (uint16_t)((dma_idx << 5) | dma_type);
   // dma_type: 1 = host input, 2 = parameter_0 (ext[2]), 3 = host output
   // Example (3-op SwiGLU with W_gate, W_up, W_down concatenated in one ext[2]):
   //   c[2] input_0: 0x01 | c[4] W_gate: 0x22 | c[6] W_up: 0x42 | c[8] W_down: 0x62 | c[10] output_0: 0x83
   ```
2. **Offset Addressing Inside a Single Concatenated `ext[2]` Blob**:
   - Every parameter `0xa40` chunk receives the **same** base `parameter_0` IOVA via `DIVE` `PutBits` at bit offset `37`.
   - Sub-blob byte offset `off_bytes` (16-byte aligned) is encoded directly in the VLIW DMA descriptors:
     - `0x940`: `flit[0].lo[47:32] = (off_bytes >> 4) & 0xFFFF`
     - `0x980`: `flit[0].lo[47:32] = (off_bytes >> 3) & 0xFFFF`
3. **Eliminating Intermediate DRAM Round-Trips & MSB Flips**:
   - Keep intermediate activations in **unsigned `uint8` (`zp + 128`) Tile NM SRAM** (`768 KiB / core` -> holds up to `M=16, d_ff=1536` = `24 KiB` with `32x` headroom!):
     - **`GEMM1 -> GEMM2`**: Skip `GEMM1`'s Stage-3 MSB-flip + output `0x9c0`/`0x400`/`0xa40` AND skip `GEMM2`'s input `0xa40` + Stage-1 MSB-flip. Directly issue `0x940(W2) -> 0xa40(W2) -> 0x980(W2) -> 0x440 -> 0x500 -> CSR_040_2`.
     - **SwiGLU (`down_proj(SiLU(gate_proj(x)) * up_proj(x))`)**:
       1. `gate_proj(x)`: GEMM1 -> Mode-A `0x640` SiLU (continuous `u_gate = SiLU(x)/s_g + zp_g + 128`, NO MSB flip!) -> NM SRAM `buf_A`.
       2. `up_proj(x)`: GEMM2 reusing `x` in NM SRAM -> NM SRAM `buf_B`.
       3. Elementwise `mul`: VICA binary `0x100`/`0x200` (`0x56 = 0x100e00`, verified in `vec_mul_1x16x16x16`) -> `buf_C = buf_A * buf_B`.
       4. `down_proj(buf_C)`: GEMM3 -> Stage-3 MSB-flip -> single `0xa40` DRAM store!
   - **Projected Latency Impact**: Collapses **4 VII calls -> 1 VII call** per FFN block (saving `3 * ~0.9 ms = ~2.7 ms/layer`, i.e., **~80 ms/token on 30-layer SmolLM2-135M**!).

---

## 4. Clean-Room C++17 Porting Guide for ``libopenbuenos``

```cpp
namespace openbuenos {

// 1. CSR 0x56 encoding (25-bit)
static constexpr uint32_t kCsr56Int8Zp128 = (0x80u << 13) | (0u << 4); // 0x0100000 (proven live)
static constexpr uint32_t kCsr56Int8Zp8   = (0x08u << 13) | (0u << 4); // 0x0010000 (probe 1)
static constexpr uint32_t kCsr56Int4Zp8   = (0x08u << 13) | (7u << 4); // 0x0010070 (compiler W4A8; saturates on A0)

// 2. Pack 208-byte 0x640 LoadCoefficientTables packet (exact bit layout)
inline void encode_0x640(uint8_t out[208],
                         const uint8_t hdr_10b[10], // bits [75:0] preserved from base chunk[5]
                         const float coeffs[8][5],
                         const float bp[7],
                         uint16_t degree_tail = 0x7804) {
  std::memset(out, 0, 208);
  auto put_bits = [&](int bit_off, uint64_t val, int width) {
    for (int b = 0; b < width; ++b) {
      if ((val >> b) & 1ULL) out[(bit_off + b) >> 3] |= (uint8_t)(1u << ((bit_off + b) & 7));
    }
  };
  // Preserve bits [75:0]
  for (int b = 0; b < 76; ++b) {
    if ((hdr_10b[b >> 3] >> (b & 7)) & 1u) out[b >> 3] |= (uint8_t)(1u << (b & 7));
  }
  int idx = 0;
  for (int s = 0; s < 8; ++s)
    for (int k = 0; k < 5; ++k, ++idx) {
      uint32_t u; std::memcpy(&u, &coeffs[s][k], 4);
      put_bits(76 + 32 * idx, u, 32);
    }
  for (int j = 0; j < 7; ++j, ++idx) {
    uint32_t u; std::memcpy(&u, &bp[j], 4);
    put_bits(76 + 32 * idx, u, 32);
  }
  put_bits(76 + 32 * 47, degree_tail, 16); // bits [1595:1580] = 0x7804 (deg-4) or 0x7801 (deg-1)
}

// 3. Patch chunk[5] HDR_DATA (+272..+320) for Mode-A zero-overhead GEMM + SiLU/GELU
inline void patch_c5_hdr_for_fused_act(uint8_t* hdr_320b,
                                       int zp_mid, float s_mid,
                                       float fclip_min, float fclip_max) {
  auto set_bits = [&](int rel_bit, uint32_t val, int width) {
    uint8_t* p = hdr_320b + 272;
    for (int b = 0; b < width; ++b) {
      int pos = rel_bit + b;
      if ((val >> b) & 1u) p[pos >> 3] |= (uint8_t)(1u << (pos & 7));
      else                 p[pos >> 3] &= (uint8_t)~(1u << (pos & 7));
    }
  };
  auto f2u = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
  set_bits(80,  (uint32_t)((zp_mid + 128) & 0xFF), 8);
  set_bits(189, f2u(s_mid), 32);
  set_bits(222, f2u(fclip_max), 32);
  set_bits(259, f2u(fclip_min), 32);
  set_bits(328, 0u, 8); // zp_out_hw = 0 (zp_out + negative-wrap folded into 0x640 c0)
}

// 4. Pack per-channel scale + optional bias tail (32B/64B stride records for 9-chunk 0x240)
inline std::vector<uint8_t> pack_pc_tail(int N, const float* eff_scales, const int32_t* bias) {
  int n4 = (N + 3) & ~3;
  std::vector<uint8_t> out;
  auto push_u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
  };
  for (int j = 0; j < n4 / 4; ++j) {
    if (eff_scales) {
      for (int q = 0; q < 4; ++q) {
        float v = (4 * j + q < N) ? eff_scales[4 * j + q] : 0.0f;
        uint32_t u; std::memcpy(&u, &v, 4); push_u32(u);
      }
      out.insert(out.end(), 16, 0);
    }
    if (bias) {
      for (int q = 0; q < 4; ++q) push_u32((4 * j + q < N) ? (uint32_t)bias[4 * j + q] : 0u);
      out.insert(out.end(), 16, 0);
    }
  }
  return out;
}

} // namespace openbuenos
```

---

## 5. Live Silicon Probe Results & `W8A16 (BF16)` Resolution

All probes were executed directly on live `buenos_a0` silicon via the zero-Google-`.so` bare-metal runner (`maps_has_google_tpu_libs == false`). See [`../hardware-errata.md`](../hardware-errata.md) and [`google-ai-edge/LiteRT#12178`](https://github.com/google-ai-edge/LiteRT/issues/12178) for full errata analysis.

| Probe | Purpose | Live Silicon Outcome (`buenos_a0`) |
|---|---|---|
| `probe_i4_1x16x16_csr56_0010000.dgc0` | Clears `0x56[6:4]` (`7 -> 0`), keeps `ext[2]=w+8` & `zp_w=8` | **PASS**: `15/16` exact, `max_abs_err = 1` (matches INT8 ref) |
| `probe_i4_1x16x16_csr56_0010070_nibpack.dgc0` | Keeps `0x56=0x0010070`, feeds true 4-bit nibble-packed `ext[2]` | **DIAGNOSED**: saturated `[127]*16` (`err=130`) — `0x56[6:4]=7` fails on nibbles |
| `fused_gemm_silu_1x16x16_zp0.dgc0` | Mode-A `+0 B` Fused `GEMM + SiLU` (`zp_out=0`, single-tile) | **PASS**: `16/16` exact (`max_abs_err = 0`, `hw_us=370`) |
| `fused_gemm_gelu_1x16x16_zp0.dgc0` | Mode-A `+0 B` Fused `GEMM + GELU` (`zp_out=0`, single-tile) | **PASS**: `16/16` exact (`max_abs_err = 0`, `hw_us=434`) |
| `fused_gemm_silu_1x16x16_zpm112.dgc0` | Mode-A `+0 B` Fused `GEMM + SiLU` (asymmetric `zp_out=-112`) | **PASS**: `16/16` exact (`max_abs_err = 0`, `hw_us=564`) |
| `fused_gemm_silu_4x32x32_zp0.dgc0` | Mode-A `+0 B` Fused `GEMM + SiLU` (`4x32x32` 16-tile scatter) | **PASS**: `128/128` exact (`max_abs_err = 0`, `hw_us=570`) |
| `probe_dequant_i4_nibpack_1x64x32.dgc0` | Stock `P=1024 B` 4-bit nibble-packed `dequant_i4_fc` (`CSR 0x56=0x0000026`, `0x980 w8=0x02200000`) + `_w8zero` | **TIMEOUT**: `poll(resp_efd)` timed out identically (`P=1024 B` under-delivers into `1:2` `u8->BF16` `0x980`) |
| `probe_deq_b2b_w8_on_i8_128B.dgc0` | Working `1x16x16` + `0x980 w8=0x02200000` + `P=128 B` (`128B -> 256B` exact slot fit) | **PASS (`hw_us=398`, `16/16` bit-exact `max_abs_err=0` vs `BF16(float(u8 - 8))`)**: causally proves `0x980 w8=(1<<21)\|(zp<<22)` is **`1:2` `UINT8 -> BFLOAT16` (`1 B -> 2 B`)**, NOT 4-bit nibble unpack |
| `probe_deq_b4a_u8_2048B.dgc0` | `dequant_i4_fc` expanded from `P=1024B` nibbles to `P=2048B` unpacked `u8 (w+8)` + `128B` `FP32` scales at `+0x800` | **PASS (`hw_us=426`): `32/32` BIT-EXACT `FP32` (`max_abs_err = 0.0`)** |
| `probe_deq_b4b_u8_2048B_c5_to_c7.dgc0` | Same `P=2048B u8` payload with `c[5]->c[7]` scale ordering | **PASS (`hw_us=411`): `32/32` BIT-EXACT `FP32` (`max_abs_err = 0.0`)** |

---

## 6. Epistemic Status Ledger

- **`[OBSERVED]`** `18/18` bit-exact `ext[2]` reconstruction across the compiled `phase5_sweep_int4` suite: unpacked `w+8` (`16/16` W4A8 FC, `0%` BW savings), 32B/64B strided per-channel FP32 scale & bias tails (`6/6`), and 4-bit nibble packing (`dequant_i4_fc_1x64x32`).
- **`[OBSERVED]`** `11/11` bit-exact 7-chunk `c0..c6` reconstruction (INT8 twin + 7 per-tensor INT4 + 3 blockwise INT4 proving `BlockwiseQuantization` is silently dropped to `s_w=1.0`), plus `6/6` exact 9-chunk W4A8 `0x240` (`0x61[19:18]`) and `0xa40`/`0x940`/`0x980` offset verification.
- **`[CAUSALLY_PROVEN]`** Live `buenos_a0` silicon parity (zero Google `.so`s): `12,288/12,288` exact (`max_abs_err=0`) on standalone splines, `4/4` Mode-A `+0 B` fused `GEMM + SiLU/GELU` probes `max_abs_err=0`, `probe_i4_1x16x16_csr56_0010000` PASS (`15/16` exact, `max_abs_err=1`), and `probe_i4_1x16x16_csr56_0010070_nibpack` saturated `[127]*16`.
- **`[CAUSALLY_PROVEN]`** Live silicon ablation of `0x980 w8` (`probe_deq_b2b_w8_on_i8_128B`, `hw_us=398`): `w8 = (1<<21) | (zp<<22)` implements **`1:2` hardware `UINT8 -> BFLOAT16` (`BF16(float(u8 - zp))`, `16/16` exact, `max_abs_err=0`)**, **NOT** 4-bit nibble unpack. Tensor G5 (`buenos_a0`) has no native 4-bit hardware DMA/systolic unpack — `Q4_0` / `W4` weights should be unpacked once at load time into cached `uint8` DMA-BUFs (same `1 B/weight` DMA bandwidth as `Q8_0`).
- **`[CAUSALLY_PROVEN]`** Deadlock fix ([`google-ai-edge/LiteRT#12178`](https://github.com/google-ai-edge/LiteRT/issues/12178)) & **Native `W8A16 (BF16)` FP32-I/O GEMM** unlock: expanding `1x64x32` `dequant_i4_fc` from `1,024 B` nibbles to `2,048 B u8 (w+8)` + `128 B FP32` scales executes natively in **`411–426 us`** with **`32/32` bit-exact `FP32` outputs (`max_abs_err = 0.0`)**.
