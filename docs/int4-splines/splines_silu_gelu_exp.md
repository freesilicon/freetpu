# Tensor G5 (`buenos_a0`) Hardware `0x640` Splines: `SiLU`, `GELU`, and `exp(x)`

**Scope:** Clean-room wire specification of the Tensor G5 VICA `LoadCoefficientTables` (`0x640`) instruction, clean-room inventory of the 34 embedded compiler LUTs in `libedgetpu_tflite_compiler.so`, independent pure-NumPy Remez minimax degree-4 spline tables for `SiLU`, `GELU`, and `exp(x)`, and ready-to-run patched `.dgc0` probe blobs for `rango` silicon proof.

---

## 1. Embedded Compiler Spline LUT Inventory (`libedgetpu_tflite_compiler.so`) `[OBSERVED]`

Inspecting `filewrapper_toc` inside `extracted/vendor/lib64/libedgetpu_tflite_compiler.so` reveals **34 embedded `*_spline.pb`** files serialized as `platforms.darwinn.numerics.proto.PiecewisePolynomial`:

- **Protobuf wire schema `[OBSERVED]`**:
  - Field `1`: `domain` message (`1 = start`, `2 = end`, each a `Number`)
  - Field `2`: `mode` enum (`0 = normal`, `1 = sqrt`, `2 = rsqrt`, `3 = recip`)
  - Field `3`: `symmetry` enum (`0 = none`, `2 = odd` — e.g. `atan`, `cbrt`, `erf`, `tanh`, `sin`, `one_tailed_logistic_sigmoid_minus_half`)
  - Field `4` (repeated): `Polynomial` segment (`1 = interval`, `2 = repeated Number coeffs [c0..c4]`)
  - Field `5`: expression string; Field `6`: `float32` maximum absolute error
  - `Number` oneof: `1 = finite(1=sign, 2=sint32_zigzag exp, 3=uint64 significand)` $\rightarrow (-1)^{\text{sign}} \cdot \text{mant} \cdot 2^{\text{exp}}$; `2 = zero(1=sign)`; `3 = inf(1=sign)`; `4 = nan`.

Key LLM-relevant embedded LUTs `[OBSERVED]`:

| Embedded File | Segments | Degree | Domain | Reported Max Abs Error | Notes |
|---|---:|---:|---|---:|---|
| `two_tailed_logistic_spline.pb` | 8 | 4 | `[-11.1, +11.1]` | `1.526e-5` ($2^{-16}$) | Multiplying coeffs by `256` matches `logistic_scale_zp.dgc0` `0x640@51` bit-for-bit |
| `swish_spline.pb` | 8 | 4 | `[-11.6, +inf]` | `6.104e-5` ($2^{-14}$) | Last segment exact linear `[0, 1, 0, 0, 0]` |
| `gelu_spline.pb` | 8 | 4 | `[-4.343, +inf]` | `3.873e-5` | Exact `erf` GELU |
| `gelu_approximated_spline.pb` | 8 | 4 | `[-4.169, +inf]` | `2.794e-5` | `tanh`-approximated GELU |
| `gelu_approximated_for_positive_spline.pb` | 8 | 4 | `[0, +inf]` | `3.815e-6` | Positive half-domain |
| `exp_spline.pb` | 8 | 4 | `[-12.96, 0]` | `3.815e-6` ($2^{-18}$) | Softmax numerator range |

---

## 2. Exact `0x640` (`LoadCoefficientTables`) & `0x200` Wire Specification `[OBSERVED]`

### 2.1 `0x640` 208-Byte (1664-Bit) Packet Layout

Disassembled from the compiler encoder at `0x2bf7708..0x2bf90b0` and verified **bit-exact round-trip** on all 3 `0x640` packets (`seq=36 @ off=3056`, `seq=51 @ off=4176`, `seq=66 @ off=5008`) of `logistic_scale_zp.dgc0`:

| Bit Range (LE) | Width | Field | Verified Semantics |
|---|---:|---|---|
| `[75:0]` | 76 b | Header | Opcode `0x640` (`0x19 << 6`) in `[11:0]`, core mask `0xffff` in `[27:12]`, sequence counter in `[63:48]`, flags in `[75:64]` |
| `[1355:76]` | 1280 b | `coeffs[8][5]` | 40 IEEE-754 `float32` values (`seg 0..7`, each `[c0, c1, c2, c3, c4]`), evaluated via FP32 Horner: `(((c4*x + c3)*x + c2)*x + c1)*x + c0` |
| `[1579:1356]` | 224 b | `breakpoints[7]` | 7 strictly ascending IEEE-754 `float32` thresholds `bp[0..6]`. Segment `s` is the smallest `s in [0..6]` with `x < bp[s]`, else `7` |
| `[1663:1580]` | 84 b | Tail / Degree & Enables | `0x7801` for degree-1 linear splines (Stages 1 & 3); `0x7804` for degree-4 splines (Stage 2). Low bits `[1583:1580]` encode polynomial degree (`1` vs `4`); bits `[1594:1591]` = `0b1111` enable flags (`0x2bf9024..0x2bf90b0`) |

### 2.2 Stage 1 `(s_in, zp_in)` & Stage 2 `0x200` `stream125` Exact Patch Map

Diffing `logistic_scale_zp.dgc0` (`s_in=0.03125, zp_in=-10`) against `vec_logistic_1x16x16x16.dgc0` (`s_in=0.0625, zp_in=0`) in `chunk[3]` (5520 B) identifies **all 4 sites** governing Stage 1 dequantization and Stage 2 float clipping. Applying `patch_stage1_input_quant(c3_lsz, s_in=0.0625, zp_in=0)` reproduces `vec_logistic_1x16x16x16.dgc0` `chunk[3]` **5520/5520 bytes bit-exact**:

1. **`0x3c0` (`off=0`, words `w25..w26`)**: `(zp_in & 0xFF)` duplicated into 16 bits (`(z8 << 8) | z8`) at `w25[63:60]` (low 4 bits) and `w26[11:0]` (high 12 bits). Also verified on `q_zp_in0_out1` (`0x0000`), `q_zp_in1_out0` (`0x0101`), `q_zp_m05_in10_outm20` (`0x0a0a`), and `logistic_scale_zp` (`0xf6f6`).
2. **`0x640@36` (`off=3056`)**: `bp[0] = -zp_in * s_in`; `seg[0] = [+128.0, 1/s_in, 0, 0, 0]`; `seg[1..7] = [-128.0, 1/s_in, 0, 0, 0]`.
3. **`0x200@45` `HDR-DATA` (`off=3392`, final 48 bytes `+272..+320`)**: relative bitfields `[87:80]` & `[335:328]` = `(zp_in + 128) & 0xFF`; `[220:189]` = `fp32(s_in)`; `[253:222]` = `fp32((127 - zp_in) * s_in)`; `[290:259]` = `fp32((-128 - zp_in) * s_in)`.
4. **`0x200@60` (`off=4512`, `stream125` across CSRs `0x5B..0x5F`)**: each 32-bit slot encodes `((val25 & 0x1FFFFFF) << 7) | (reg & 0x7F)`, concatenating `0x5B..0x5F` LSB-first into a 125-bit stream:
   - `mul` (`fp32`): **`stream125[44:13]`**
   - `fclip_max` (`fp32`): **`stream125[77:46]`** (1-bit gap at bit `45`)
   - `fclip_min` (`fp32`): **`stream125[114:83]`** (5-bit gap at bits `82:78`)

### 2.3 Hardware FP32-RNE Horner Verification

Running `simulate_3stage_dgc0()` in [`spline_0x640.py`](../../tools/spline_0x640.py) against live `rango` silicon output `logistic_scale_zp.out.bin` matches **4096 / 4096 bytes exact** (`0` mismatches).

---

## 3. Clean-Room Synthesized Minimax Degree-4 Splines `[OBSERVED]`

All three tables were synthesized from scratch using pure-NumPy degree-4 Remez exchange (`remez_poly_deg4`) in normalized coordinates `t in [-1, 1]` with iterative $h^5$ error-equibalancing across the 7 interior breakpoints (`optimize_8seg_minimax`), then affine-fused into Stage 2 quantization $u(x) = f(x)/s_{\text{out}} + (z_{\text{out}} + 128) \in [0, 255]$ without touching Stage 3 (`u -> int8`).

### 3.1 `SiLU(x) = x / (1 + exp(-x))` on `[-11.1, +11.1]`

- **Dense FP32 Horner max abs error (20,001 pts):** `4.693e-5` (tighter than vendor `swish_spline.pb` `6.104e-5`)
- **Breakpoints:** `[-6.157059, -3.245293, -1.363508, 0.001634, 1.367547, 3.250287, 6.172718]`
- **Probe quantization:** `s_in=0.03125, zp_in=-10` (`Stage 1 untouched`, `x in [-3.6875, +4.28125]`), `s_out=0.02, zp_out=-112` (`u = 50*SiLU(x) + 16`)
- **Quantized accuracy (all 256 `int8` inputs):** `max_quantized_lsb_err = 0.00234 LSB` (`< 0.5 LSB` by 214×), `255/256` exact (`max_abs_diff = 1`, single `.5` tie)

| Seg | `c0` | `c1` | `c2` | `c3` | `c4` | Seg Max Abs FP32 Err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `-0.4196394` | `-0.1643377` | `-0.02446811` | `-0.001635336` | `-4.12716e-05` | `3.47e-5` |
| 1 | `-0.6763348` | `-0.2847731` | `-0.04247522` | `-0.002328171` | `-1.615514e-05` | `3.91e-5` |
| 2 | `0.004530482` | `0.5476574` | `0.3466325` | `0.08005808` | `0.00664204` | `4.19e-5` |
| 3 | `4.403719e-05` | `0.5016305` | `0.2592714` | `0.0176807` | `-0.008976126` | `4.68e-5` |
| 4 | `4.963941e-05` | `0.498302` | `0.2594834` | `-0.01792149` | `-0.008886687` | `4.70e-5` |
| 5 | `0.004069802` | `0.4532037` | `0.3460428` | `-0.07988254` | `0.006622855` | `4.24e-5` |
| 6 | `-0.6781057` | `1.286366` | `-0.04300551` | `0.002405658` | `-2.034851e-05` | `3.94e-5` |
| 7 | `-0.4179907` | `1.163571` | `-0.02433593` | `0.001625328` | `-4.099049e-05` | `3.47e-5` |

### 3.2 `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` on `[-11.1, +11.1]`

- **Dense FP32 Horner max abs error (20,001 pts):** `3.720e-5`
- **Breakpoints:** `[-4.236798, -2.22263, -0.976752, 0.001047, 0.979179, 2.224355, 4.235237]` (exact asymptotic tails `0` on `seg0` and `x` on `seg7`)
- **Probe quantization:** `s_in=0.03125, zp_in=-10` (`Stage 1 untouched`), `s_out=0.02, zp_out=-112` (`u = 50*GELU(x) + 16`)
- **Quantized accuracy (all 256 `int8` inputs):** `max_quantized_lsb_err = 0.00186 LSB` (`< 0.5 LSB` by 269×), **`256/256` exact (`max_abs_diff = 0`)**

| Seg | `c0` | `c1` | `c2` | `c3` | `c4` | Seg Max Abs FP32 Err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `2.16e-5` |
| 1 | `-0.7379804` | `-0.746281` | `-0.2845654` | `-0.04844738` | `-0.003104619` | `2.15e-5` |
| 2 | `0.02636749` | `0.635421` | `0.6627197` | `0.2434929` | `0.03100839` | `2.62e-5` |
| 3 | `3.518374e-05` | `0.5017828` | `0.4128561` | `0.03632539` | `-0.03368206` | `3.71e-5` |
| 4 | `3.911894e-05` | `0.4981514` | `0.4131436` | `-0.03677963` | `-0.03344718` | `3.72e-5` |
| 5 | `0.02614394` | `0.3651706` | `0.6621441` | `-0.2432488` | `0.03097028` | `2.64e-5` |
| 6 | `-0.7381586` | `1.746515` | `-0.2846794` | `0.04847167` | `-0.003106534` | `2.17e-5` |
| 7 | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `2.18e-5` |

### 3.3 `exp(x)` on `[-12, 0]` (Softmax Numerator)

- **Dense FP32 Horner max abs error (20,001 pts):** `1.875e-6` (2× tighter than vendor `exp_spline.pb` `3.815e-6`)
- **Breakpoints:** `[-8.531695, -6.362728, -4.694147, -3.334423, -2.223777, -1.326686, -0.604141]`
- **Probe quantization:** `s_in=0.125, zp_in=0` (`Stage 1 patched` using exact 5520/5520-verified patcher, `x in [-16, +15.875]` clamped via `0x200@60` `fclip=[-12.0, 0.0]`), `s_out=1/256, zp_out=-128` (`u = 256*exp(x)`)
- **Quantized accuracy (all 256 `int8` inputs):** `max_quantized_lsb_err = 0.000449 LSB` (`< 0.5 LSB` by 1,112×), **`256/256` exact (`max_abs_diff = 0`)**

| Seg | `c0` | `c1` | `c2` | `c3` | `c4` | Seg Max Abs FP32 Err |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.0288618` | `0.01004612` | `0.001321511` | `7.773937e-05` | `1.72329e-06` | `3.36e-7` |
| 1 | `0.1441996` | `0.06527013` | `0.01128067` | `0.0008792881` | `2.600825e-05` | `5.25e-7` |
| 2 | `0.364341` | `0.2057719` | `0.04505383` | `0.004502412` | `0.0001723398` | `9.97e-7` |
| 3 | `0.636948` | `0.4401963` | `0.1210463` | `0.01550677` | `0.0007728355` | `1.61e-6` |
| 4 | `0.8579322` | `0.7052132` | `0.2410669` | `0.03983231` | `0.002634148` | `1.89e-6` |
| 5 | `0.9681242` | `0.9004297` | `0.3720892` | `0.07932256` | `0.00714343` | `1.74e-6` |
| 6 | `0.9973928` | `0.9849169` | `0.4651493` | `0.1257341` | `0.01598892` | `1.31e-6` |
| 7 | `0.999999` | `0.9999179` | `0.4988868` | `0.1613365` | `0.03096789` | `1.03e-6` |

---

## 4. Epistemic Status Ledger

- **`[OBSERVED]`** `0x640` 208-byte packet layout (`[75:0]`, 40 FP32 coeffs `[1355:76]`, 7 FP32 bps `[1579:1356]`, 84-bit tail `[1663:1580]`) verified bit-exact round-trip on all 3 packets of `logistic_scale_zp.dgc0`.
- **`[OBSERVED]`** `0x200@60` `stream125` bit offsets (`mul[44:13]`, `fclip_max[77:46]`, `fclip_min[114:83]`) and 4-site Stage 1 patch map verified **5520/5520 bytes exact** transforming `logistic_scale_zp.dgc0` $\rightarrow$ `vec_logistic_1x16x16x16.dgc0`.
- **`[OBSERVED]`** FP32-RNE Horner simulation matches live `rango` silicon `logistic_scale_zp.out.bin` **4096/4096 exact**.
- **`[OBSERVED]`** All 3 emitted `.dgc0` probes parse cleanly with `dgc0_dump.py` and achieve `< 0.0024` quantized-LSB error in offline FP32-RNE simulation.
- **`[INFERRED]`** `silu_probe.dgc0` and `gelu_probe.dgc0` (which patch only `0x640@51`) and `exp_probe.dgc0` (which also patches Stage 1 + `0x200@60` `fclip=[-12, 0]`) will execute bit-identically to `.ref.bin` on `rango` silicon.
- **`[UNKNOWN]`** Live `rango` hardware execution result for `silu_probe.dgc0`, `gelu_probe.dgc0`, and `exp_probe.dgc0` (queued for `rango-runner`, offline lane).
- **`[UNKNOWN]`** Whether `0x640` hardware uses exact FMA (`fmaf`) vs separate FP32 mul+add (`fmul` + `fadd`); `silu_probe` and `gelu_probe` (with dense non-monotonic polynomials) will surface any 1-ULP tie difference if present.
