# Tensor G5 (`laguna` / `buenos_a0`) Vector, Quantization & Activation ISA

Executive summary and live hardware sweep results for `isa-analyzer-vec` (`rango` offline clean-room analysis; all hardware execution via `the on-device runner`).

See [vector_alu_isa.md](vector_alu_isa.md) for full bitfield tables, VLIW packet formats, `stream125` register layouts, `0x640` spline coefficients, and the Observed / Inferred / Unknowns ledger.

---

## Executive Summary

1. **25-bit CSR Slot Packing (`0x200` / `0x100` / `0x500`)**:
   Every nonzero 32-bit CSR slot packs `(val25 << 7) | reg7` (`bit 7` is `val25[0]`, explaining `0xdc..0xe1` register bytes), with header `reg 0x50` / `0x70` storing `(num_csr_writes << 5) | 0x10`.
2. **FP32 Requantize & Activation Clamps (`stream125` across `0x5b..0x5f`)**:
   - `stream125[44:13]`: IEEE-754 **`float32` multiplier** (Stage 1: quantized scale ratio `M1`; Stage 2: real output scale `s_out`).
   - `stream125[77:46]`: IEEE-754 **`float32` upper clamp** (`fclip_max`: Stage 1 `qmax - zp_out`, `RELU6` `+96.0f` = `0x42c00000` -> `0x5d=0x160000`; Stage 2 `(qmax - zp_out) * s_out`).
   - `stream125[114:83]`: IEEE-754 **`float32` lower clamp** (`fclip_min`: `0.0f` for `RELU`/`RELU6`, `(-128 - zp_out)` Stage 1 / `(-128 - zp_out)*s_out` Stage 2 for `NONE`).
   - `reg 0x57`: `uint8` `((zp_out + 128) & 0xff) << 3`. `reg 0x61`: `val24 = 0x100 + 2*zp_out` (`FC` sets bit 17 `0x020100 + 2*zp_out`).
3. **Native IEEE-754 FP32 Round-Half-to-Even (`np.rint`)**:
   Hardware computes FP32 RNE rather than TFLite's RHU or fixed-point integer `(m, shift)`, verified to `1 ULP` (`0x3fc00001` on `q_s_nonpot_1_5` and live patch `p1_q_s1_mul_05_to_15`: `4096/4096`). On `ADD`, `zp_out` is added **after** RNE (`add_nonpot`: `4096/4096`).
4. **Vector Loop Sharing & Fused Activations**:
   `0x5c0` (`208 B`) and `0x4c0` (`160 B`) are invariant VICA engine/thread setup packets across `RELU`, `QUANTIZE`, `ADD`, `MUL`, `LOGISTIC`, and `FC`; ops differ via `0x640` LUTs, `0x200`/`0x100` CSRs, and `HDR-DATA` (`320 B`). `FC` (`0x500`, `fc_act_{none,relu,relu6}`) fuses `RELU`/`RELU6` at `+0 B` vs `+480 B` standalone `chunk[5]` in `pipe2_fc_relu_1x16x16`.
5. **`0x640` Affine Sign-Split & `LOGISTIC` Degree-4 Spline**:
   `0x640` packs 47 `float32` words (`bit 76`, stride `32`). Affine stages encode `bp0 = -zp * s` splitting `+/-128` at slope `1/s`; `LOGISTIC` adds a middle 8-segment degree-4 minimax sigmoid spline (`16/16` and `4096/4096` exact match).
6. **Tiling & `finish_and_sleep` Tail**:
   Elementwise drain uses `32` elements/core across up to `16` cores (`512` elem/pass) via `0x400(112B)` + `0x400(8B)` in `chunk[last]`, ending in `2 x (0x680+0x900+0x180)` + `0x900` (`0x000fffff4000<m>900`) + `0x3e0` (`0x00000000008003e0`).

---

## Key Artifacts & Scripts

- Spec: [vector_alu_isa.md](vector_alu_isa.md)
- Batch 1b Generator & Manifest: `gen_vec_sweep.py`, `manifest.json`
- Dissection & Solver Scripts: `decode_vliw_blocks.py`, `analyze_b0_details.py`, `analyze_b1b.py`, `solve_csr_bitfields.py`, `solve_remaining_bits.py`, `solve_exact_remaining.py`, `solve_final_details.py`
- Causal Live Patch Builder & Verifier: `build_patch_proof.py`, [verify_patch_proof.py](../../tools/verify_patch_proof.py)

---

## Live `rango` Sweep Results

### 1. Causal `--run-dgc0` Patch Proofs (`5/5` Executed)

Summary: `summary.json`

| Probe | Base `.dgc0` -> Patched Bitfield | `hw_us` | Host Formula Exact Match | Causal Finding |
|---|---|---:|---:|---|
| `p1_q_s1_mul_05_to_15` | `q_s_0_5` `c3 0x200@58` `0x5b..5c` `0.5` -> `0x3fc00001` (`1.5+1ULP`) | `137` | **`4096/4096`** | Confirms `0x5b..5c` sets Stage 1 multiplier incl. LSB `+1 ULP` (`-51 -> -77`, `25 -> +38`) |
| `p2_q_s2_mul_half_x2` | `q_s_0_5` `c3 0x200@73` `0x5c` `0.0625` -> `0.125` (`2x`) | `106` | **`4096/4096`** | Proves Stage 2 `0x5b..5c` executes at runtime after Stage 1 RNE (`q_mid * 0.125 * 16.0`) |
| `p3_mul_nonpot_s1_to_05` | `mul_nonpot` `c5 0x100@100` `0x5b..5c` `0.06667` -> `0.5` | `117` | **`4096/4096`** | Proves `MMA_CSR_WRITE` (`0x100`) `0x5b..5c` sets `MUL` Stage 1 product multiplier |
| `p4_add_relu_to_clip_m128_96` | `add_act_relu` `c5 0x200@98` `0x5d` `127->96`, `0x5f` `0->-128` | `96` | **`4096/4096`** | Proves `0x5d` (`fclip_max`) and `0x5e..5f` (`fclip_min`) live-switch activation bounds (`[-128, +96]`) |
| `p5_q_zp_out_1_to_10` | `q_zp_in0_out1` `c3 0x200@60` `0x61` `0x102->0x114` (`+1->+10`), `0x5d` `0x7e000->0x75000` | `124` | `3943/4096` (`-10..-2` -> `-1`) | Proves `0x61 = 0x100 + 2*zp_out` (`+9` on valid lanes) + `0x640` `bp0=-0.0625` negative-segment cap |

### 2. Batch 1b — Quantized Vector/Activation Differential Sweep (`33/33` Pass)

Summary: `summary.json` (`compile=33/33`, `execute=33/33`, `verify(err<=1)=33/33`, `dgc0_pulled=33/33`, `out_dump_pulled=33/33`).

| Model | Op | Bytes | `hw_us` | `mean_ms` | TFLite `ref` Exact | Model Match (`f32_rne` / Spline) |
|---|---|---:|---:|---:|---:|---:|
| `q_s_1_0` | `QUANTIZE` | `647680` | `66` | `15.72` | `4096/4096` (`err=0`) | `4096/4096` |
| `q_s_0_5` | `QUANTIZE` | `663040` | `105` | `16.20` | `3076/4096` (`err=1`) | `4096/4096` |
| `q_s_0_25` | `QUANTIZE` | `663040` | `104` | `15.95` | `3076/4096` (`err=1`) | `4096/4096` |
| `q_s_0_125` | `QUANTIZE` | `663040` | `105` | `11.89` | `3569/4096` (`err=1`) | `4096/4096` |
| `q_s_2_0` | `QUANTIZE` | `663040` | `108` | `16.60` | `4096/4096` (`err=0`) | `4096/4096` |
| `q_s_nonpot_0_625` | `QUANTIZE` | `663040` | `117` | `13.10` | `3842/4096` (`err=1`) | `4096/4096` |
| `q_s_nonpot_0_333` | `QUANTIZE` | `663040` | `97` | `16.76` | `2736/4096` (`err=1`) | `4096/4096` |
| `q_s_nonpot_1_5` | `QUANTIZE` | `663040` | `138` | `19.13` | `3365/4096` (`err=1`) | **`4096/4096`** (`f64=3382`) |
| `q_zp_in1_out0` | `QUANTIZE` | `662464` | `118` | `14.00` | `4096/4096` (`err=0`) | `4096/4096` |
| `q_zp_in0_out1` | `QUANTIZE` | `662464` | `115` | `14.07` | `4096/4096` (`err=0`) | `4096/4096` |
| `q_zp_in_m128_out127` | `QUANTIZE` | `662464` | `160` | `15.53` | `4096/4096` (`err=0`) | `4096/4096` |
| `q_zp_m05_in10_outm20` | `QUANTIZE` | `663040` | `123` | `12.02` | `3076/4096` (`err=1`) | `4096/4096` |
| `relu_1x1x1x16` | `RELU` | `653824` | `85` | `13.83` | `16/16` (`err=0`) | `16/16` |
| `relu_1x1x1x64` | `RELU` | `653952` | `103` | `19.80` | `64/64` (`err=0`) | `64/64` |
| `relu_1x64x1x64` | `RELU` | `659648` | `105` | `21.86` | `4096/4096` (`err=0`) | `4096/4096` |
| `quant_1x1x1x16` | `QUANTIZE` | `653760` | `92` | `32.43` | `12/16` (`err=1`) | `16/16` |
| `quant_1x1x1x64` | `QUANTIZE` | `653952` | `85` | `14.34` | `48/64` (`err=1`) | `64/64` |
| `add_1x1x1x16` | `ADD` | `656704` | `124` | `17.11` | `8/16` (`err=1`) | `16/16` |
| `add_1x1x1x64` | `ADD` | `656768` | `88` | `15.03` | `46/64` (`err=1`) | `64/64` |
| `add_base_0625` | `ADD` | `668288` | `95` | `16.21` | `4096/4096` (`err=0`) | `4096/4096` |
| `add_act_relu` | `ADD` | `668288` | `109` | `12.30` | `4096/4096` (`err=0`) | `4096/4096` |
| `add_act_relu6` | `ADD` | `668288` | `102` | `13.45` | `4096/4096` (`err=0`) | `4096/4096` |
| `add_asym_scale` | `ADD` | `668288` | `128` | `15.29` | `3119/4096` (`err=1`) | `4096/4096` |
| `add_nonpot` | `ADD` | `668288` | `110` | `14.36` | `3629/4096` (`err=1`) | **`4096/4096`** (`rint+zp`) |
| `mul_base_0625` | `MUL` | `667968` | `99` | `14.47` | `3891/4096` (`err=1`) | `4096/4096` |
| `mul_act_relu` | `MUL` | `667968` | `96` | `12.86` | `3954/4096` (`err=1`) | `4096/4096` |
| `mul_act_relu6` | `MUL` | `667968` | `97` | `14.41` | `3974/4096` (`err=1`) | `4096/4096` |
| `mul_nonpot` | `MUL` | `667968` | `134` | `13.95` | `4075/4096` (`err=1`) | `4096/4096` |
| `logistic_1x1x1x16` | `LOGISTIC` | `654080` | `130` | `24.56` | `16/16` (`err=0`) | `16/16` (spline) |
| `logistic_scale_zp` | `LOGISTIC` | `663296` | `118` | `17.93` | `4096/4096` (`err=0`) | `4096/4096` (spline) |
| `fc_act_none` | `FULLY_CONNECTED` | `656064` | `104` | `13.08` | `15/16` (`err=1`) | `—` |
| `fc_act_relu` | `FULLY_CONNECTED` | `656064` | `100` | `13.15` | `15/16` (`err=1`) | `—` |
| `fc_act_relu6` | `FULLY_CONNECTED` | `656064` | `113` | `14.51` | `16/16` (`err=0`) | `—` |

### 3. Batch 0 (`vec_*`/`pipe2_*`) & Batch 1 (`s=1, zp=0` Unquantized Baseline)

- **Batch 0** (`summary.json`): `vec_add_1x16x16x16` (`668288 B`, `162 us`), `vec_add_1x64x1x64` (`662912 B`, `204 us`), `vec_mul_1x16x16x16` (`667968 B`, `198 us`), `vec_relu_1x16x16x16` (`663040 B`, `157 us`, `4096/4096`), `vec_logistic_1x16x16x16` (`663296 B`, `166 us`, `4096/4096`), `vec_quantize_requant_1x16x16x16` (`663040 B`, `163 us`, `err=1`), `pipe2_fc_relu_1x16x16` (`656512 B`, `247 us`, `16/16`), `pipe2_conv1x1_add_1x16x16x16` (`670272 B`, `170 us`, `err=2`), `pipe2_add_logistic_1x16x16x16` (`669120 B`, `191 us`, `err=4`).
- **Batch 1** (`summary.json`): Initial unquantized (`s=1.0, zp=0`) `.tflite`s compiled & executed `19/19` (`RELU`/`ADD`/`MUL`/`LOGISTIC`/`FC`), proving CSR omission behavior (`0x5c`/`0x5d` omitted when `s=1.0`); `14/14` `QUANTIZE` models failed `MLIR TFL -> DWC Conversion` due to missing `QuantizationParameters` (resolved in Batch 1b).
