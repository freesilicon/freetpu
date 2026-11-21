# Tensor G5 (`buenos_a0`) Native `INT4` (`W4A8`) Offline RE & Sweep Specification

**Scope:** Offline reverse engineering of `extracted/vendor/lib64/libedgetpu_tflite_compiler.so` (82.8 MB) and `libedgetpu_litert.so` for 4-bit weight (`TensorType_INT4`) acceptance rules, hardware `ext[2]` parameter packing, subbyte/subchannel compiler lowering passes, and the 19-model `sweep_int4` differential `.tflite` pack queued for `rango`.

---

## 1. TFLite Frontend & Delegate Acceptance Matrix `[OBSERVED]`

### 1.1 `libedgetpu_litert.so` Delegate Filter Rules

| Address | Observed Check | Implication |
|---|---|---|
| `0x3da2dc` | `filter->type == data_type \|\| data_type == kTfLiteInt16 \|\| filter->type == kTfLiteInt4` | Delegate accepts `FULLY_CONNECTED` with `INT4` weights paired with `INT8` (`W4A8`) or `INT16` (`W4A16`) activations |
| `0x53448c` | `input->type in {kTfLiteInt8, kTfLiteInt16}` and `filter->type in {kTfLiteInt8, kTfLiteInt4}` | Confirms `W4A8` and `W4A16` FC delegation |
| `0x41ca50` | `filter->type != kTfLiteInt4` in sparse quantized FC | Sparse quantized FC explicitly rejects `INT4` weights |
| Filter check | `Blockwise embedding lookup only supports Int4 and Int2 data` (`Float32`/`Float16` outputs) | Delegate supports Blockwise in `EMBEDDING_LOOKUP`, not in `FULLY_CONNECTED` |

### 1.2 Compiler Importer & Validation Rules

| Address | Observed Behavior / String | Implication |
|---|---|---|
| `FlatBufferToMlir` | Maps `TensorType_INT4` (`17=0x11`) $\rightarrow$ `i4`, `UINT4` (`20=0x14`) $\rightarrow$ `ui4`, `INT2` (`21=0x15`) $\rightarrow$ `i2` | Native MLIR `i4` tensor import supported |
| `GetQuantizedType` | Rejects `details_type == 1` (`CustomQuantization` $\rightarrow$ `"Cannot handle experimental quantization"`); `scale.size() == 1` $\rightarrow$ `UniformQuantizedType`; `scale.size() > 1` $\rightarrow$ `UniformQuantizedPerAxisType` along `quantized_dimension` | Per-tensor and per-channel `INT4` quant imported directly (`BlockwiseQuantization` `details_type == 2` is dropped / falls back to subchannel patterns) |
| `0x31b5650..0x31b9000` | `ConvertTFLFullyConnectedOp`: requires `weights_format == "DEFAULT"` (`0x31b81c8`), static shapes, 2D constant filter, 1D constant signed `i32` bias (or omitted); checks `output_int4_allowed` (`0x31b566c`) | Matches standard TFLite `FULLY_CONNECTED` v10 (`INT4`) |
| `0x3133de8` | `"Filter element's storage type is not i4 or i8."` | Both `i4` and `i8` filter storage accepted in FC lowering |
| `0x34fd3c4` | `"INT4 input is not supported on the current chip generation."` | Activations **cannot** be `INT4` (`W4A4` rejected); weights are `INT4` (`W4A8` / `W4A16`) |
| `0x34fd530` | `"Elementwise (CwiseBinaryOp) op does not support 4-bit types for inputs."` | Elementwise ops reject 4-bit inputs |
| `0x0fea1c` | `tfl.dequantize` accepts `QI4` (`tensor of QI8 type or QI4 type or QUI8 type or QI16 type or QUI16 type values`) + `FuseBatchMatMulDequantizeOp` | Weight-only `DEQUANTIZE(INT4) -> FC` / `BatchMatMul` path exists in MLIR |
| `0x138ec88` | `parameter_sub_channel_block_size == kAllowedParameterSubChannelBlockSize` (`mov w1, #0x20` $\rightarrow$ **`32`**) | Hardware subchannel parameter block size is strictly **32** |
| `0x2f2a990` | `subchannel_activation_block_size == 32` | Activation subchannel block size is also **32** |
| `0x265ccd8` | `"Y!=1 is not supported for FullyConnectedSubChannel layer"` | Subchannel FC is restricted to `Y == 1` |
| `0x2bd2ddc` | `"Subchannel scale mode is not supported for int32 output data type."` | Subchannel FC cannot output raw `int32` |
| `0x2669610` | `"Subbyte data type conversion is not supported for < 4 bit for this hardware configuration."` | `INT2` (`< 4 bit`) rejected on `buenos_a0`; **`INT4` is supported** |
| `0x3b2c114` | `"For the case of SubByte ui4 and ui2, %s tensor size = %d * bits_per_element needs to be multiple of 8"` | Total packed `ui4` tensor size must be byte-aligned (even element count) |

---

## 2. Hardware `ext[2]` Parameter Packing Layout `[OBSERVED]`

### 2.1 Signed-to-Unsigned Offset (`+8`)

Disassembling `ActivationRange` (`0x4a4da54`) and its `.rodata` range tables (`0x5e5698` / `0x4df8e0`) proves:
- `FIXED_POINT4` (`enum = 3`) has range **`[0, 15]`** (`min = 0, max = 15`), and the compiler's `DataType` enum (`0x5e6c56`) has **`FIXED_POINT4`** with **no `SIGNED_FIXED_POINT4`** variant (unlike `SIGNED_FIXED_POINT8` / `SIGNED_FIXED_POINT16` / `SIGNED_FIXED_POINT32`).
- Confirmed by string `0x287da4`: `zero_point <= numerics::ActivationRange(config::FIXED_POINT4).max`.
- Therefore, signed TFLite `TensorType_INT4` (`[-8, +7]`, `zp = 0`) is stored in hardware `ext[2]` with **`+8` unsigned offset** (`u4 = (w_s4 + 8) & 0x0F`, hardware zero-point `8`), exact 4-bit analog of `FIXED_POINT8` (`+128` offset).

### 2.2 Nibble Packing Order (`CopyElement` @ `0x2bd20c0..0x2bd2820`)

Disassembly of `CopyElement` (`0x2bd2400..0x2bd2480`) proves **little-endian nibble order**:
- **Even element `2k`** $\rightarrow$ **low nibble `byte[k] & 0x0F`** (`bits [3:0]`)
- **Odd element `2k + 1`** $\rightarrow$ **high nibble `(byte[k] >> 4) & 0x0F`** (`bits [7:4]`)
- This matches the TFLite `TensorType_INT4` buffer format identically (only adding the `+8` unsigned bias per nibble when repacking into `ext[2]`).

### 2.3 Subchannel Scale + Weight Tile Interleaving (`0x2bd2108..0x2bd2494`)

When subchannel scales are present (`x21 != 0` in `CopyElement`):
- Allocates `scales_bytes + weights_bytes` contiguously in `ext[2]`.
- Copies `min(w29, 32)` scale bytes immediately preceding each `min(w27, 32) x min(w29, 32)` transposed 4-bit weight tile (`PackMimdParametersForAnySubByteGraph` @ `0x10ca39`, logging `#parameters, unpadded_z_out_dim, z_in_dim, ... is_native_subbyte_graph, num_elements_per_byte, is_subchannel_graph, num_dequantization_tiles` @ `0x1b1b86` / `0x368fe44`).

---

## 3. Compiler Lowering & VLIW Opcode Architecture (`0x040` / `0x440` / `0x500`)

### 3.1 Observed Compiler Passes `[OBSERVED]`

Disassembly and pass metadata in `libedgetpu_tflite_compiler.so` reveal **two distinct 4-bit weight execution mechanisms** in the DarwiNN compiler:

1. **Software / VICA Unpack Path (`sub_byte_param_unpack_buffer`)**:
   - Replaces subbyte constants with a constant layer using `fixed8` types.
   - Uses dedicated SRAM buffer role `sub_byte_param_const_buffer` $\rightarrow$ unpacks via logical shift/mask into `sub_byte_param_unpack_buffer` as `fixed8` before feeding the standard INT8 systolic array (`0x040`/`0x500`).
2. **Narrow-to-Wide DMA Path (`0x980`)**:
   - `LowerSubChannelParameterScales` adds a sync-guarded subchannel scaling pass.
   - `NarrowToWide` DMA (`0x440` + `0x980 w8`) supports `INT8_TO_BFLOAT` (`1:2` `uint8 -> BF16`) on `buenos_a0`; live silicon ablation shows `buenos_a0` has no native 4-bit hardware nibble-unpack unit (see [`../hardware-errata.md`](../hardware-errata.md)).
   - TensorOp dispatches `0x040`/`0x100`/`0x200`/`0x1040` with per-axis/subchannel sync descriptors.

### 3.2 Resolution Summary

Live silicon ablation on `buenos_a0` resolved both questions ([`../hardware-errata.md`](../hardware-errata.md)):
1. Stock `INT4` `FULLY_CONNECTED` unpacks weights offline to `u8 (w + 8)` at `1 B/weight`.
2. Descriptor `0x980 w8` implements `1:2` `UINT8 -> BFLOAT16` (`BF16(float(u8 - zp))`), enabling native `W8A16 (BF16)` FP32-I/O GEMM; TFL3 `BlockwiseQuantization` falls back to `s_w = 1.0` unless unpacked/lowered as subchannel weights.

**Resolution Strategy:** `sweep_int4` explicitly pairs **`fc_i8_pt_1x16x16_id_ref`** (INT8) and **`fc_i4_pt_1x16x16_id`** (INT4) with **identical weights (`4*I`), identical scales, identical inputs, and bit-identical `.ref.bin` outputs** — a single `dgc0_dump.py` diff on `rango`'s compiled `.dgc0` pair will expose every single `ext[2]`, `0x040`, `0x440`, and `0x500` bitfield immediately.

---

## 4. Pre-Built 19-Model `sweep_int4` Pack `[OBSERVED]`

Generated by [`gen_int4_sweep.py`](../../tools/gen_int4_sweep.py) into `work/int4-splines/sweep_int4/`; all 19 `.tflite` files pass strict FlatBuffer validation (`TFL3`, `ceil(N*K/2)` INT4 byte buffers, `BlockwiseQuantization` table integrity, exact CPU reference `.ref.bin`):

| # | Model Name | Weights / Act | Quant Mode | `M x K x N` | Purpose |
|---:|---|---|---|---|---|
| 1 | `fc_i8_pt_1x16x16_id_ref` | `int8` / `int8` | per-tensor | `1 x 16 x 16` | **Exact INT8 twin** of #2 (`ref.bin` identical) for 1-to-1 ISA bit-diff |
| 2 | `fc_i4_pt_1x16x16_id` | `int4` / `int8` | per-tensor | `1 x 16 x 16` | Scaled-identity `INT4` (`w=4*I`) — isolates `ext[2]` `+8` & nibble order |
| 3 | `fc_i4_pt_1x32x16` | `int4` / `int8` | per-tensor | `1 x 32 x 16` | Full `[-8, +7]` range, `K=32` |
| 4 | `fc_i4_pt_1x64x32` | `int4` / `int8` | per-tensor | `1 x 64 x 32` | Full range, `K=64, N=32` |
| 5 | `fc_i4_pt_1x128x32_bias` | `int4` / `int8` | per-tensor | `1 x 128 x 32` | Non-zero `int32` bias |
| 6 | `fc_i4_pt_1x256x64_bias` | `int4` / `int8` | per-tensor | `1 x 256 x 64` | Multi-tile LLM projection size |
| 7 | `fc_i4_pt_4x64x32` | `int4` / `int8` | per-tensor | `4 x 64 x 32` | Batched `M=4` |
| 8 | `fc_i4_pt_8x64x32` | `int4` / `int8` | per-tensor | `8 x 64 x 32` | Batched `M=8` (prefill) |
| 9 | `fc_i4_pt_1x32x16_zp_relu` | `int4` / `int8` | per-tensor | `1 x 32 x 16` | Asymmetric `zp_in=-10, zp_out=-128` + fused `RELU` |
| 10 | `fc_i4_pt_1x32x16_nobias` | `int4` / `int8` | per-tensor | `1 x 32 x 16` | Bias-free LLM linear projection (`inputs=[0, 1]`) |
| 11 | `fc_i4_pc_1x32x16` | `int4` / `int8` | per-channel | `1 x 32 x 16` | Per-channel `scales[16]`, `quantized_dimension=0` |
| 12 | `fc_i4_pc_1x64x32_bias` | `int4` / `int8` | per-channel | `1 x 64 x 32` | Per-channel `scales[32]` + bias |
| 13 | `fc_i4_pc_1x128x64_bias` | `int4` / `int8` | per-channel | `1 x 128 x 64` | Per-channel `scales[64]` + bias |
| 14 | `fc_i4_pc_4x128x64` | `int4` / `int8` | per-channel | `4 x 128 x 64` | Per-channel batched `M=4` |
| 15 | `fc_i4_blk32_1x128x32` | `int4` / `int8` | `blockwise_bs32` | `1 x 128 x 32` | TFL3 `BlockwiseQuantization` `block_size=32` (`kAllowedParameterSubChannelBlockSize`) |
| 16 | `fc_i4_blk64_1x128x32` | `int4` / `int8` | `blockwise_bs64` | `1 x 128 x 32` | TFL3 `BlockwiseQuantization` `block_size=64` |
| 17 | `fc_i4_blk128_1x256x64` | `int4` / `int8` | `blockwise_bs128` | `1 x 256 x 64` | TFL3 `BlockwiseQuantization` `block_size=128` |
| 18 | `fc_i4_w4a16_pc_1x64x32` | `int4` / `int16` | per-channel | `1 x 64 x 32` | Acceptance probe: `W4A16` (`int16` act × `int4` per-channel weight) |
| 19 | `dequant_i4_fc_1x64x32` | `int4` / `float32` | `dequant_pc` | `1 x 64 x 32` | Acceptance probe: `DEQUANTIZE(INT4) -> FULLY_CONNECTED` |
