# Tensor G5 (`buenos_a0`) GEMM / Systolic ISA & Requantization Spec

Clean-room recovery of the INT8 matrix-multiply / fully-connected (`FULLY_CONNECTED` / `CONV_2D`) instruction stream, parameter packing, `0x040` CSR bank, and epilogue requantization hardware semantics on Tensor G5 (`laguna` / `buenos_a0`), verified against compiled `.dgc0` artifacts and live hardware execution via `--run-dgc0` on `rango`.

All claims carry explicit epistemic status:
- `[OBSERVED]`: Directly measured in compiled `.dgc0` bitstreams or `--run-dgc0` execution JSONs.
- `[CAUSALLY_PROVEN]`: Verified by running synthesized/ablated `.dgc0` kernels directly on `rango` hardware and checking exact element-wise outputs.
- `[INFERRED]`: Structural generalization consistent with all observed sweep points, not yet tested beyond the swept domain.
- `[UNKNOWN]`: Opaque bits or multi-tile regions outside the parameterized single-tile emitter.

---

## 1. Analyzed Corpus & Hardware Coverage

1. **Sweep 1 (`work/isa-gemm/sweep1/manifest.json` -> `work/device-runner/phase3_gemm_sweep1/`)** `[OBSERVED]`:
   - `22/22` models compiled (`tfl3` -> `DGC0`), executed, and verified (`max_abs_err <= 1`) on `rango` (`hw_us = 143..246`).
   - Parameterized dimensions: `[M, K, N]` covering `M in {1, 2, 3, 4, 8}`, `K in {16, 32, 64}`, `N in {8, 16, 20}`, plus `zp_in in {-37, 0, +17}`, `zp_out in {-128, -45, 0, +29}`, power-of-two `s_out in {1/64, 0.0625, 1/256}`, non-power-of-two `s_out in {0.05, 0.03}`, `fused_activation in {NONE=0, RELU=1, RELU6=3}`, `g20_diff_wb`, and `g21_zero_bias`.
   - Every Sweep 1 model lowers to a **7-chunk** hardware sequence (`chunk[0]..chunk[6]`): for `(M=1, K=16)`, chunk byte lengths are `[320, 80, 64, 2464|2480, 64, 1456, 736]` (or `chunk[3]=2304 B` in `g21_zero_bias`).
2. **Batch 0 GEMM/Conv & Batch 1b (`work/device-runner/phase3_batch0/`, `phase3_isa_vec_b1b/`)** `[OBSERVED]`:
   - `gemm_fc_{1x16x16, 1x16x32, 1x64x64, 1x128x64, 4x32x32}` and `pipe2_fc_relu_1x16x16` (`err <= 1`, `hw_us = 136..247`); `fc_act_{none,relu,relu6}` in `phase3_isa_vec_b1b`.
   - `gemm_conv1x1_*`, `gemm_conv3x3_*` (`chunks[5]=1312..1712`, larger `chunk[3]=4496..15488` and `chunk[6]=4896..7456`).
   - `gemm_fc_int16x8_1x16x16` **failed TFLite->DGC0 compilation** (`compile.ok=false, status=13`: `invalid element type !quant.uniform<u16:f32, ...> for BiasParameterOp, reason: Only signed 32-bit fixed point is implemented`).
3. **Custom Synthesized Microkernel & Causal Probes (`work/isa-gemm/emit_custom_gemm.py` -> `work/device-runner/phase3_gemm_custom/`)** `[CAUSALLY_PROVEN]`:
   - `emit_custom_gemm.py --self-test` verifies **15/15 bit-exact** synthesis of `chunk[0..6]` (modulo 4-bit cyclic tag in `chunk[6] f041.lo[15:12]`) across `g00..g11`, `g18_k16_n8`, `g19_k16_n20`, and `gemm_fc_1x16x32`.
   - `custom_gemm_microkernel.dgc0` (`M=1, K=16, N=32, zp_in=-11, s_out=0.05, zp_out=+19, ACT_RELU`) + 4 causal probes executed directly on `rango` via `--run-dgc0`.

---

## 2. Weight & Bias Parameter Layout (`DWS0`, `ext_slots[2]`)

`[OBSERVED]` `[CAUSALLY_PROVEN]` Across all Sweep 1 and Batch 0 INT8 FC models, parameters live in `external_buffers[2]` (`DWS0` storage):

1. **Channel Padding**:
   ```text
   n_pad = ceil(N / 8) * 8
   ```
   Observed parameter sizes (`param_bytes`):
   - `g21_zero_bias` (`K=16, N=16`, zero bias): `256 B` (weights only, bias omitted)
   - `g18_k16_n8` (`K=16, N=8`): `160 B = 8*4 + 4*8*4`
   - `g00_base_1x16x16` (`K=16, N=16`): `320 B = 16*4 + 4*16*4`
   - `g19_k16_n20` (`K=16, N=20`, `n_pad=24`): `480 B = 24*4 + 4*24*4`
   - `gemm_fc_1x16x32` (`K=16, N=32`): `640 B = 32*4 + 4*32*4`
   - `g16_k32_n16` (`K=32, N=16`): `576 B = 16*4 + 8*16*4`
   - `g17_k64_n16` (`K=64, N=16`): `1088 B = 16*4 + 16*16*4`
2. **Binary Order inside `ext_slots[2]`**:
   - **Bias block** (omitted when all biases == 0, as in `g21_zero_bias`): `int32[n_pad]` little-endian; lanes `n >= N` are `0`.
   - **Weight block**: `[K // 4, n_pad, 4]` `uint8`:
     - Active lanes `n < N`: `(int8_weight[n, k4*4 + lane] + 128) & 0xFF`.
     - Padded lanes `n >= N` (`g19_k16_n20`): raw byte `0x00` (not `128`).
     - Note: zero-point bias (`+128` offset) maps signed `[-128, 127]` to `[0, 255]`; when `bias == 0`, the compiler also omits the `0x500` bias setup block (`f133` in `g00`) in `chunk[3]` (`2464 -> 2304 B`) and omits tail CSRs `0x48, 0x49, 0x4a, 0x61` (`g21_zero_bias`, `zp_out=0`).

---

## 3. Seven-Chunk Execution Pipeline (`g00_base_1x16x16` Baseline)

Each flit is 16 bytes (`<QQ` `lo, hi`, bit 0 at LSB of `lo`). Header opcode tags appear in `lo[15:0]` (`0x1xxx` descriptor headers; `0x680` sync/guard ends in `f680`).

| Chunk | `g00` Size | Stream Tags Observed (`flit:tag`) | Functional Role |
|---|---:|---|---|
| `chunk[0]` | `320 B` | Fixed `0x680` / `0x900` init sequence | Hardware window / tile bank reset & sync `[OBSERVED]` |
| `chunk[1]` | `80 B` | `f000:0x680_guard`, `f002:0x8940` | Input tensor SRAM ring & `M` dimension (`f004.lo[21:16] = 4*M`) `[OBSERVED]` |
| `chunk[2]` | `64 B` | `f000:0x9a40` (tag `0x01`) | Flexible DMA in0 (`in_bytes = M * K`) `[OBSERVED]` |
| `chunk[3]` | `2464 B` (2480 B w/ `0x5e`/extra CSR) | `f000:0x3c0`, `f014:0x440`, `f056:0x640`, `f069:0x200`, `f093:0x3c0`, `f107:0x680_guard`, `f125:0x440`, `f133:0x500`, `f142:0x040` | Input unpack + `zp_in`/`s_in` dequant + param/bias setup + systolic & epilogue `0x040` CSRs `[CAUSALLY_PROVEN]` |
| `chunk[4]` | `64 B` | `f000:0x0a40` (tag `0x22`) | Flexible DMA par (`param_bytes`) `[OBSERVED]` `[CAUSALLY_PROVEN]` |
| `chunk[5]` | `1456 B` | `f000:0x940`, `f003:0x980`, `f022:0x680_guard`, `f024:0x5c0`, `f037:0x4c0`, `f049:0x640`, `f062:0x200` | Param unpack + output mask + `1/s_out`, `-zp_out*s_out`, `s_out`, `zp_out` requantize/clamp `[CAUSALLY_PROVEN]` |
| `chunk[6]` | `736 B` | `f000:0x0a40` (tag `0x43`), `f004:0x9c0`, `f008:0x400`, `f015..f041:0x680/0x900` | Output DMA (`M * N`) + output tile `N-1` bounds + drain `[OBSERVED]` `[CAUSALLY_PROVEN]` |

---

## 4. Exact Bitfield Map (Across Chunks & `0x040` CSRs)

### 4.1 `chunk[1]` Input TTU (`M` dimension) `[OBSERVED]`
- **`f004.lo[21:16]`** (word `w4[25:16]` of `0x940` descriptor): `4 * M` (`M=1 -> 4`, `M=2 -> 8`, `M=3 -> 12`, `M=4 -> 16`, `M=8 -> 32`; confirmed across `g00, g12..g15` and matches `#42` `the DMA ISA spec` `(M << 2) + sram_off_32b`).

### 4.2 `chunk[2]`, `chunk[4]`, `chunk[6]` Flexible DMA (`0xa40`) `[CAUSALLY_PROVEN]`
- **`f003`** byte transfer count (for `bytes <= 8192`):
  - `f003` bit `[63:48]` = `(transfer_bytes - 1) * 8` (`f003.lo[63:48]`)
  - `f003` bit `[95:80]` = `(transfer_bytes - 1) * 8` (`f003.hi[31:16]`)
  - Confirmed for `chunk[2]` (`in_bytes`), `chunk[4]` (`param_bytes`, e.g. `160, 320, 480, 640`), and `chunk[6]` (`out_bytes`, e.g. `8, 16, 20, 32`).
  - General note from `the DMA ISA spec` (`#42`): full 32-bit producer/consumer `(bytes - 1)` lives at `[466:435]` and `[498:467]`, which equals `(bytes - 1) << 3` within `[63:48]` / `[95:80]` of `f003` (`bit 435 - 384 = 51` = byte `f003` bit `51` -> `(bytes - 1) * 8` at bit `48` is bit `[50:48]` low 3 bits = `0` plus `(bytes-1)` starting at bit 51!).

### 4.3 `chunk[3]` Bitfields (`M=1, K=16` offsets anchored on `g00`)
1. **`0x3c0` (`f012..f013`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - Byte `zp_in & 0xFF` is duplicated at bits `[131:124]` and `[139:132]` relative to `f012` (`f012.hi` high byte nibble + `f013.lo` low bytes; e.g. `zp_in=0 -> 0x00`, `+17 -> 0x11`, `-37 -> 0xdb`, `-11 -> 0xf5`).
2. **`0x640` (`f066`, bits `[107:76]`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - Encodes IEEE-754 `fp32(-zp_in * fp32(s_in))` (`0x00000000` when `zp_in == 0`).
3. **`0x200` Input Dequantize/Clamp Vector Block (`f086..f088`, relative bit offsets)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - Bits `[87:80]` and `[335:328]`: `(zp_in + 128) & 0xFF` (offset-binary input bias; `g00: 0x80`, `g01 (+17): 0x91`, `g02 (-37): 0x5b`).
   - Bits `[220:189]`: `fp32(s_in)`
   - Bits `[253:222]`: `fp32((127 - zp_in) * fp32(s_in))`
   - Bits `[290:259]`: `fp32((-128 - zp_in) * fp32(s_in))`
4. **`0x440` Parameter Staging (`f128`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - `p_field = (n_pad // 8) * 0x400 - 0x20` (`n_pad=8 -> 0x3e0`, `n_pad=16 -> 0x7e0`, `n_pad=24 -> 0xbe0`, `n_pad=32 -> 0xfe0`).
   - `f128` bits `[39:24] = p_field`; `[63:48] = p_field + 2`; `[79:64] = p_field << 4`.
5. **`0x040` Tail CSR Stream (`f142..end` in `M=1, K=16`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - Header flit (`0x1040` in `lo[15:0]`): `hdr_hi[23:12] = 3 + 2 * num_u32` (`44` u32s -> `0x05b`, `45` -> `0x05d`, `46` -> `0x05f`, `47` -> `0x061`).
   - Packed words: little-endian `u32 = (val25 << 7) | (csr_id & 0x7F)`, emitted in strictly ascending `csr_id` order (`0x04..0x61`), zero-padded to 16-byte flit boundary (`<= 12` data flits fits `g00`'s `8512 B` DIVE container slack, updating the chunk3 size word at `dive[chunk3_off - 4]` from `2464` to `2480`).

   **Recovered `0x040` CSR Table (`25-bit val25`)**:

   | CSR ID | Parameter | Encoding & Evidence | Status |
   |---|---|---|---|
   | `0x06` | Systolic K/M trip? | `((M * (K // 4) - 1) << 20) \| 0x2` across `M in {1,2,3,4,8}, K in {16,32}` (`0x300002..0x1f00002`, `g16: 0x700002`); `g17 (M=1, K=64)` emits `0xb00002` (step `11`, matching `0x3b=0xb000`) | `[OBSERVED]` |
   | `0x09, 0x0c, 0x0e, 0x11` | Systolic neg-strides | Scale linearly with `M * (K // 4) - 1` (`M=1..8`, `K=16,32`) | `[OBSERVED]` |
   | `0x28` | `N`-tile shape | Omitted when `N == 16`; otherwise `((N - 1) << 19) \| (N // 2 - 1)` (`N=8: 0x380003`, `N=20: 0x980009`, `N=32: 0xf8000f`) | `[OBSERVED]` `[CAUSALLY_PROVEN]` |
   | `0x2a..0x38` | 41-bit loop bounds | Concatenated 25-bit stream: 5x 41-bit `(bound=1023, step)` entries starting at stream bit `30`; low bits carry `3, 7, 11` for `K=16, 32, 64` (`entry 0`) and `M - 1` (`entry 1`: `0,1,2,3,7` for `M=1,2,3,4,8`) | `[OBSERVED]` |
   | `0x3b` | Inner `K` bound | `(3 << 12)` (`0x3000`) for `K=16`, `0x7000` (`K=32`), `0xb000` (`K=64`) | `[OBSERVED]` |
   | `0x3c, 0x3d` | Outer `M` loop | When `M > 1`: `0x3c = ((M - 1) << 12) \| 2`, `0x3d = (0x1000 - 2 * (M - 1)) & 0x1FFFFFF` (`g17 K=64` also writes `0x3c = 0x1000`) | `[OBSERVED]` |
   | `0x58` | Output row/K accum? | `0x024000` (`M=1, K=16,32`), `0x028000` (`M>1, K=16`), `0x068000` (`M=1, K=64`) | `[OBSERVED]` |
   | `0x59` | `N`-tile mask pair | Omitted when `N == 16`; otherwise `((N - 1) << 5) \| (N - 1)` (`N=8: 0xe7`, `N=20: 0x273`, `N=32: 0x3ff`) | `[OBSERVED]` `[CAUSALLY_PROVEN]` |
   | `0x5b..0x61` | **Continuous 25b epilogue bank** | **`mul_f32`, `fclip_max_f32`, `fclip_min_f32`, `zp_out`** packed continuously LSB-first across 25-bit CSRs `stream125 = sum(val25(0x5b+i) << (25*i))` (jointly verified with `the Vector ALU ISA spec` `#48`): | `[CAUSALLY_PROVEN]` |
   | `0x5b` | `mul_f32[11:0]` | `(fp32(s_in_f * s_w_f / s_out_f) & 0xFFF) << 13` (`stream125[24:13]`) | `[CAUSALLY_PROVEN]` |
   | `0x5c[19:0]` | `mul_f32[31:12]` | `(fp32(s_in_f * s_w_f / s_out_f) >> 12) & 0xFFFFF` (`stream125[44:25]`) | `[CAUSALLY_PROVEN]` |
   | `0x5c[21]` | `fclip_max_f32[0]` | Bit `46` of `stream125` (`fclip_max_f32[0]`, set to `1` only in `g08_sout_nonpow2_003` by compiler); causal `rango` test (`probe_0x5c_bit21_zero`) with bit 21 = 0 produced `16/16` exact IEEE-754 `round(acc * real_mul)` (`err<=1` vs TFLite fixed-point ref) | `[CAUSALLY_PROVEN]` |
   | `0x5d[24:12]` | `fclip_max_f32[28:16]` | `stream125[74:62] = (bf_max & 0x1FFF) << 12`, where `fclip_max_f32 = fp32_bits(float(qmax - zp_out))` | `[CAUSALLY_PROVEN]` |
   | `0x5e[2:0]` | `fclip_max_f32[31:29]` | `stream125[77:75]` (for `qmax-zp_out in [2..255]`, `bf_max >> 13 == 2`, matching reset default `0x0000002` when `0x5e` is omitted). **Causal proof**: writing `0x5e = 0x1000000` (clearing `0x5e[1]`) turns `fclip_max_f32=0x42c40000` (`98.0`) into denormal `0x02c40000` (`~0.0`), clamping all elements `> zp_out` (`29`) to `29`; restoring `0x5e[1]=1` (`0x1000002`) restores `16/16` exact output! | `[CAUSALLY_PROVEN]` |
   | `0x5e[24]` | `fclip_min_f32[16]` | `stream125[99]` (`bf_min & 1`, bit 16 of `fclip_min_f32 = fp32_bits(float(qmin - zp_out))`, with low bits `[15:0]` at `stream125[98:83] = 0x5e[23:8]`) | `[CAUSALLY_PROVEN]` |
   | `0x5f[14:0]` | `fclip_min_f32[31:17]` | `stream125[114:100] = (bf_min >> 1) & 0x7FFF` (`0x0000000` when `RELU`/`RELU6` sets `qmin == zp_out`; e.g. `-128` -> `0xc300 >> 1 = 0x6180`; `-157` -> `0xc31d` sets `0x5f=0x618e` and `0x5e[24]=1`) | `[CAUSALLY_PROVEN]` |
   | `0x61[18:2]` | `zp_out + 128` | `(1 << 18) \| (((zp_out + 128) & 0xFF) << 2)`; omitted in `g21_zero_bias` (`zp_out=0`, no bias) | `[OBSERVED]` `[CAUSALLY_PROVEN]` |

   *Crucial hardware finding on requantization arithmetic* `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   The Tensor G5 requantizer operates in **IEEE-754 float32 (round-half-to-even)**, NOT fixed-point `MultiplyByQuantizedMultiplier`. Evidence: (a) `0x5b/0x5c` carry raw `fp32(real_mul)` (`stream125[44:13]`), (b) `0x5d/0x5e/0x5f` carry full 32-bit `fp32` upper/lower clamp bounds (`stream125[77:46]` and `[114:83]`, where `act=RELU/RELU6` sets clamp bounds before `+ zp_out`), and (c) on `probe_0x5c_bit21_zero` element `n=14` (`acc=1013`, `acc * real_mul = 16.48763`), live `rango` hardware output `16` matched IEEE-754 round-half-to-even `round(acc * real_mul) = 16` rather than fixed-point TFLite `17` (also independently proved in `#48` by `the Vector ALU ISA spec`).

### 4.4 `chunk[5]` Param Unpack, Output Dequant & Dual-Tile Epilogue (`M=1, K=16` offsets)
1. **Param Unpack `0x940`/`0x980` (`f001.hi`, `f004.hi`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - With `P = param_bytes`:
     - `f001.hi[31:0] = (((-P + 4) & 0xFFFF) << 16) | (P // 4 - 8)`
     - `f004.hi[31:0] = (((-2*P + 8) & 0xFFFF) << 16) | (P // 2 - 15)`
   - Verified across `N in {8, 16, 20, 32}` and `K in {16, 32, 64}` (`P in {160, 320, 480, 576, 640, 1088}`).
2. **Output Dimension Masks `0x5c0`/`0x4c0` (`f027, f028, f037, f041, f043, f046`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - `f027`: bits `[46:35] = N` (`N << 3` in `lo[63:32]`); with `tile_n = min(N, 16)`, bits `[79:76] = tile_n - 1`, bits `[116:96] = (-4 * tile_n + 4) & 0x1FFFFF` (`0x1fffe4` for `N=8`, `0x1fffc4` for `N>=16`).
   - `f037`: bits `[86:77] = N` (`N * 0x2000` in `hi`).
   - Single partial sub-tile `N <= 8` additionally adjusts `f028 = (0xFF9000003FFFC800, 0x0000FFFF2000007F)` and masks in `f041, f043, f046`.
3. **Output Scale Block `0x640` (`f049..f061`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - `8` replicated lanes (`lane = 0..7`) starting at bit `108` with `160`-bit stride write `fp32(1.0 / fp32(s_out))`.
   - `f059` bits `[107:76]` (`10 * 128 + 76` within `f049..f061`): `fp32(-zp_out * fp32(s_out))` (`0` when `zp_out == 0`).
4. **`0x200` Tile Routing & Output Clamp/Requant Vector (`f062..f083`)** `[OBSERVED]` `[CAUSALLY_PROVEN]`:
   - `f062..f076`: exact spatial/multicast tile masks for single-tile (`N=8`, `N=16`) vs dual-tile (`N=20`, `N=32`).
   - `f079..f081` (relative bits from `f079` byte 0):
     - `[87:80]` and `[335:328]`: `(zp_out + 128) & 0xFF`
     - `[220:189]`: `fp32(s_out)`
     - `[253:222]`: `fp32((127 - zp_out) * fp32(s_out))`
     - `[290:259]`: `fp32((-128 - zp_out) * fp32(s_out))`

### 4.5 `chunk[6]` Output DMA & Drain (`M=1`) `[OBSERVED]` `[CAUSALLY_PROVEN]`
- `f003`: `(out_bytes - 1) * 8` at `[63:48]` and `[95:80]`.
- `f006`: `(N - 1) * 0x84` at `[63:48]`.
- `f007`: `(out_bytes // 2)` at `[31:16]`.
- `f011`: `(N - 1) * 0x20` at `[63:48]`, `(N - 1) * 0x200` at `[79:64]`.
- `f041.lo[15:12]`: 4-bit compiler-assigned cyclic tag (`g00..g21` cycle without affecting execution semantics; masked in bit-exact self-test).

---

## 5. Container & Security Boundary Observations

1. **DIVE & DWS0 Shape Patching (`1x16x16` shell -> `1x16x32` flagship)** `[CAUSALLY_PROVEN]`:
   - `g00` (`656064 B`) already reserves 16 B alignment slack inside `ext_slots[0]` (`8512 B` DIVE payload), allowing `chunk[3]` to grow `2464 -> 2480 B` by updating `dive[chunk3_off - 4] = 2480` without relocating subsequent chunks.
   - `dive[0x2010]` stores `uint32(N_dim)`.
   - `DWS0` header stores `out_bytes` (`M * N`) at file offsets `0x1cf8`, `0x1d00`, and `0x1fa0` (verified on `N=32`, expanding blob `656064 B -> 656384 B`).
2. **No Blocking VLIW Hash Gate on Direct `--run-dgc0`** `[OBSERVED]`:
   - Modifying VLIW `chunk[3,4,5,6]`, `DIVE`, `DWS0` shapes, and `ext_slots[2]` without recomputing `DGC0` hashes (`0xf88` / `0x1c70`) succeeded cleanly on `rango` via `--run-dgc0` (`5/5` executions `execute.ok=true`, `used_sq_name=""`).

---

## 6. Unknowns Ledger (`[UNKNOWN]`)

1. **`chunk[5]` `0x200` (`f062..f076`) General `N` Formula**: Exact bitfield equations are verified for `N in {8, 16, 20, 32}`; arbitrary unaligned `N > 32` or multi-tile `N = 64` (`chunks[5]=1536 B`) uses table/empirical masks from those 4 cases rather than a closed-form generator across arbitrary `N`.
2. **`0x040` Non-`M=1, K=16` CSR Bank Generation**: For `M > 1` (`g12..g15`), `chunk[3]`, `chunk[5]`, `chunk[6]` grow substantially (`3056..5680 B`, `2752..3712 B`, `2080..3616 B`) and `0x040` expands with per-row SRAM loop entries (`0x13..0x24`, `0x3c..0x46`); `emit_custom_gemm.py` currently targets `M=1, K=16, N in {8,16,20,32}`.
3. **`CONV_2D` (`1x1` / `3x3`) Spatial Im2Col / Halo Packing**: Observed `chunk[3]=4496..15488 B` and `chunk[6]=4896..7456 B` in Batch 0, not yet parameterized.
4. **Compiler `0x5c[21]` Semantics**: Identified via `#48` (`stream125[46] = fclip_max_f32[0]`); why the compiler set LSB bit 0 of `fclip_max_f32` only on `g08_sout_nonpow2_003` is a compiler quirk, and `probe_0x5c_bit21_zero` proved bit 21 = 0 produces exact `round(acc * real_mul)` (`16/16`).
5. **Dual Role of `0x640`/`0x200` Output Vector vs `0x040` Epilogue CSRs**: For single-op `FC`, causal proof (`0x5e[2:0]`) shows `0x040` (`0x5b..0x61`) governs runtime requantization/clamping; per `#48`, `0x640` carries sign-split `int8<->uint8` / piecewise-polynomial ramp parameters (`slope = 1/s`, `bp0 = -zp*s`, `seg0/1 = +/-128`) used in vector stages. Exact hardware arbitration across fused `pipe2` stages remains bounded per-op.
