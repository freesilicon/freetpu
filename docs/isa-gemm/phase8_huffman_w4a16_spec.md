# Phase 8 Bandwidth & Precision Specification: Hardware Huffman, Sub-Byte Remapping (`W4A8`/`W4A16`), and Large-Tile `W8A16(FP16)` on Tensor G5 (`buenos_a0`)

**Date:** 2026-11-21  
**Target Silicon:** Pixel 10 (`rango`), Tensor G5 EdgeTPU (`buenos_a0`, `chip_type = 24`), prod firmware `1.1 925724637`, `PM=5`  
**Compiler Oracle Analyzed:** `device compiler shared library`  
**Verification Script:** `offline verification harness: verify_phase8_huffman_w4a16.py` (`ALL CHECKS PASSED`)  
**Live Silicon Runs:** Wave 1 (`live device suite: wave1_oracle/summary.json`, `MAPS_GOOGLE=0`, `PRIVACY_HITS=0`) and Wave 2 (`live device suite: wave2_huffman/summary.json`, `MAPS_GOOGLE=0`, `PRIVACY_HITS=0`)

---

## 1. Executive Summary & Verdicts

Phase 8 investigated four hardware/compiler levers to determine whether Tensor G5 (`buenos_a0`) can beat the `27.96 GB/s` INT8 DMA ring ceiling for LLM decode (`M=1`) or deliver higher precision / prefill throughput (`M=64`) without CPU activation quantization and outlier peeling:

| Lever | Compiler Support (`libedgetpu_tflite_compiler.so`) | Live `buenos_a0` (`rango`, `PM=5`) Verdict | Bandwidth / Throughput | Precision vs FP32 | Milestone 8B Action |
|---|---|---|---|---|---|
| **Hardware Huffman (`FlexibleDma.Huffman` / `0x0cc0`)** | Implemented in `HuffmanCompressionPass` (`0x2963800`) & `EncodeHuffman` (`0x3cd3570`), gated behind unused `buenos_a0_multimedia_huffman_config` (`ScalarCore.288=1`) | **`[CAUSALLY_PROVEN]` INACTIVE / UNIMPLEMENTED ON SILICON.** HIB ignores `bit499=1` and `fields 7..9`, reads raw uncompressed bytes from `field5@37`, and silently skips `opcode 0x0cc0`. | Cannot beat `27.96 GB/s` INT8 ceiling (`h5`/`h6` SMMU read fault past compressed `ext[2]`). | N/A (not decoded) | **DROP.** Do not build Huffman weight packing for `buenos_a0`. |
| **`SubbyteConstantRemapper` (`W4A8` INT4 weights)** | `0x2513e68` unpacks 4-bit constants to 8-bit `fixed8` (`u8 ∈ [0..15]`) in the compiler before DMA. | **`[CAUSALLY_PROVEN]` 0% DRAM SAVINGS.** `w4a8_pc_1x1536x1536` emits full `2,371,584 B` (`1 B/wt`), runs `98 us` (`24.23 GB/s` `ext[2]`). | Same as INT8 `W8A8` (`1 B/wt`). | Standard W8A8 | **DROP.** Keep `Q4_0`/`Q8_0` NEON unpacking into INT8 `ext[2]`. |
| **`W4A16` (`INT4` + `0x980` narrow-to-wide)** | Bypasses `SubbyteConstantRemapper`, halves `ext[2]` (`1.18 MB`), sets `0x980 u32[16] = 0x0220_0000` (`zp_w=8`), **still `bit 21 = 1` (`INT8_TO_BFLOAT` 1:2, NOT 1:4)**. | **`[CAUSALLY_PROVEN]` BROKEN (NO 1:4 HW UNPACK).** `0x1c3e71` asserts only `INT8_TO_BFLOAT`; zero `INT4_TO_BFLOAT` support in ISA/compiler. | N/A (halved beat count under-delivers half of weights). | Corrupted (Phase 5 confirmed) | **DROP.** No native 4-bit MAC unpack exists on `buenos_a0`. |
| **Large-Tile `W8A16` (`0x980 u32[16]=0x2020_0000`)** | Fully supported (`DEQUANTIZE+FC` and `HYBRID FC` emit **identical 9/9 chunks**; per-tensor emits **7 chunks**). | **`[CAUSALLY_PROVEN]` WORKS LIVE & IS NATIVE `FP16` (10-BIT MANTISSA) + `FP32` ACCUMULATOR — NOT `BF16`!** | Decode `1x2048`: `233 us` (`18.07 GB/s`); `1x1536` PT: `138 us` (`17.19 GB/s`); **Prefill `64x1536`: `589 us` (`512.7 GOPS`)**. | **`relL2 = 1.88e-4`** (PT, `FP32` acc) / **`2.80e-4`** (PC, `FP16` mid) — **5.7–8.2× tighter than BF16!** | **ADOPT SELECTIVELY** for outlier layers (`down_proj`, `lm_head`, Layer 0–1) & high-accuracy `M=64` prefill via **7-chunk PT (`scale=1`) + NEON per-channel scale epilogue**. |

### Key Strategic Takeaway for Milestone 8B
1. **The `27.96 GB/s` INT8 DMA ring ceiling on `buenos_a0` is a hard physical floor for 1-byte-per-weight decode.** Neither hardware Huffman nor native 4-bit DMA can compress parameter DRAM traffic on shipping Pixel 10 (`buenos_a0`) silicon. **W8A8 (`Q8_0`/`Q4_0` + Overlapped NEON)** remains the fastest decode path (`23.9–25.8 GB/s` single-tile, `27.96 GB/s` ring-pipelined vs `16.7–18.1 GB/s` for `W8A16`).
2. **Major Discovery — `W8A16` is Native IEEE `FP16` (10-bit mantissa) + `FP32` Accumulator:** Prior Phase 5 probes used exact integer activations where `FP16 == BF16 == FP32`. Wave 1 proves `buenos_a0` unpacks `u8` weights (`w_i8 ^ 0x80`) into `FP16`, converts FP32 activations to `FP16`, and **accumulates in full `FP32`** (`relL2 = 2.76e-7` vs `fp32(fp16(x) @ W) * s` on per-tensor `pt_1x1536x1536`).
3. **Prefer 7-Chunk Per-Tensor `W8A16` + CPU NEON Scale Epilogue over 9-Chunk Per-Channel `W8A16`:** Per-channel `W8A16` writes an intermediate `FP16` SRAM tensor between `conv` and `mul` (`103,456/103,456` elements have 13 zero mantissa LSBs), creating an **`FP16` overflow hazard (`> 65,504`)** on unscaled outlier dot products (`K=8960` `down_proj`). The 7-chunk per-tensor graph writes the `FP32` accumulator directly (`138 us` vs `142 us`, `relL2 = 1.88e-4` vs `2.80e-4`, zero overflow risk, zero scales DMA).

---

## 2. Epistemic Ledger

| Claim | Status | Primary Evidence |
|---|---|---|
| `buenos_a0_config.binarypb == v7_config.binarypb` (`7,562 B`); `buenos_a0_multimedia_huffman_config.binarypb` (`7,568 B`) differs in **only 3 `ScalarCore` fields** (`283: 5->8`, `287: 0->1`, `288: 0->1`); `buenos_b0_config.binarypb` keeps `{283:5, 287:0, 288:0}` | `[OBSERVED]` | Extracted MD5-verified configs in `offline verification harness: configs/`; verified by `verify_phase8_huffman_w4a16.py` check 1 |
| `buenos_a0_multimedia_huffman_config_memfile` (`0x3bdfc34`) has 0 `.text` callers; prod FW `1.1 925724637` has 0 `huffman`/`codebook` strings; `CompileTfliteFlatbuffer2` (`0xd94b70`) never sets `CompilerOptions.field_313` | `[OBSERVED]` | Disassembly of `libedgetpu_tflite_compiler.so`, string scan of `device production firmware images`, Wave 1 `opts_probe_all_bools.bin` identical to `opts_default.bin` |
| Exact hardware Huffman codec (`EncodeHuffman 0x3cd3570`, `BuildCanonicalCodes 0x4bdb38c`, `672 B` codebook `0x3cd462c`, `6 B`/256B metadata `0x3cd41d0`, `min_block=64` guard `0x3cd3e9c`, `threshold=224`) compresses Q4-like weights `2.261×`–`2.262×` | `[OBSERVED]` | `offline verification harness: huffman_codec.py`; verified bit-exact round-trip by `verify_phase8_huffman_w4a16.py` check 2 |
| Exact 6-flit (`96 B`) `FlexibleDma` (`0x0a40`) Huffman ISA layout (`bit 499` enable, `bit 500` `pad_to_256`, DIVE `fields 7/8/9` at `501/565/629`, uncompressed `size_0`/`size_1` at `435/467`) | `[OBSERVED]` | Disassembly of `0x2be9a2c..0x2bebb20`, `GetFieldBitOffset 0x3edfb90`, `0x2abdb0c..0x2abe210`, `0x254f300..0x254f368`, VLOG `0x10da6f` |
| **`buenos_a0` silicon does NOT execute `FlexibleDma.Huffman` or `HuffmanDma` (`0x0cc0`)** | `[CAUSALLY_PROVEN]` | Live `rango` Wave 2: `112/112` exact bytes (`7×16/16`, `max_err=0`) match raw `field5@37`; `h4` zero-poisoned `field5` with valid `fields 7..9` returns `[-112]*16` (exact zero-weight output); `h5`/`h6` trigger SMMU read faults (`write=0 not mapped`) reading uncompressed `2.36 MB`/`4.19 MB` past compressed `ext[2]`; `h7` ignores `0x0cc0`. Verified by `verify_phase8_huffman_w4a16.py` check 3 |
| `SubbyteConstantRemapper` (`0x2513e68`) expands 4-bit `W4A8` weights to 8-bit `u8 ∈ [0..15]` (**0% DRAM savings**) | `[CAUSALLY_PROVEN]` | Disassembly of `0x2513e68..0x251441c` + live Wave 1 `w4a8_pc_1x1536x1536` (`ext[2]=2,371,584 B`, `hw_med=98 us`, `24.23 GB/s`, unique weight bytes `[0..15]`) |
| `W4A16` `0x980` (`u32[16]=0x0220_0000`, `zp_w=8`) still runs `1:2` `INT8_TO_BFLOAT` (`bit 21 = 1`) with halved beat counts | `[OBSERVED]` | Exact chunk diff `w4a16_deq_i4_1x1536x1536.dgc0` vs `w8a16_deq_i8_1x1536x1536.dgc0` + `0x1c3e71` assertion; verified by check 4 |
| `buenos_a0` `W8A16` uses **IEEE `FP16` (10-bit mantissa) inputs + `FP32` accumulator**, universal **`[N/32, K/2, 32, 2]`** `ext[2]` packing (`w_i8 ^ 0x80`), `DEQUANTIZE+FC == HYBRID FC` (9/9 chunks), and hits `512.7 GOPS` at `64x1536x1536` | `[CAUSALLY_PROVEN]` | Live `rango` Wave 1 (`6/6` `sgs=1`, `PM=5`): `103,456/103,456` (100%) per-channel elements have 13 zero LSBs (`32/32` zero-ULP exact on `1x64x32`); per-tensor `relL2 = 2.76e-7` vs `fp32(fp16(x)@W)*s`; `6/6` byte-exact `ext[2]` match. Verified by check 4 |

---

## 3. Hardware Huffman Weight Decompression (`HibDma.FlexibleDma.Huffman` / `0x0cc0`)

### 3.1 Embedded `TargetConfig` Protobuf Diff

All 14 embedded `.binarypb` configs were extracted from `libedgetpu_tflite_compiler.so` into `offline verification harness: configs/`. Comparing `buenos_a0_config.binarypb` (`== v7_config.binarypb`, `7,562 B`, MD5 `392349c557d8...`) against `buenos_a0_multimedia_huffman_config.binarypb` (`7,568 B`, MD5 `12b5b9aa6fb2...`) and `buenos_b0_config.binarypb` (`7,559 B`, MD5 `2de94f98b7af...`) reveals that **every top-level field outside `TargetConfig.field_2` (`ScalarCore`, `TcParseTable @ 0x04ec86e0`) is byte-identical**, and `ScalarCore` differs in **exactly 3 fields**:

| `TargetConfig.ScalarCore` Field | Struct Offset | Type | `buenos_a0` / `v7` | `buenos_a0_multimedia_huffman` | `buenos_b0` | Hardware / Compiler Meaning |
|---:|---|---|---|---|---|---|
| **283** | `+0x1c0` | `int32` | `5` | **`8`** | `5` | HIB DMA channel count (sole reader `@ 0x2584ef4`). |
| **287** | `+0x10d` | `bool` | `0` | **`1`** | `0` | Enables `MultimediaDma` (`opcode 0x32` -> `0x0c80`). |
| **288** | `+0x10e` | `bool` | `0` | **`1`** | `0` | Enables `FlexibleDma.Huffman` (`0x0a40` 6-flit) & `HuffmanDma` (`0x33` -> `0x0cc0`). |

Key compiler/firmware gating facts (`[OBSERVED]`):
- `buenos_a0_multimedia_huffman_config_memfile` (`0x3bdfc34`) has **zero callers** in `.text`; it is dead/bringup-only data inside the shipped Android vendor library.
- Even `buenos_b0_config.binarypb` keeps `{283: 5, 287: 0, 288: 0}` — indicating Huffman HIB hardware was not enabled in `buenos_a0` or `buenos_b0` production configs.
- `CompileTfliteFlatbuffer2` (`0xd94b70`) unconditionally selects `v7_config` (`field_288 = 0`) for `chip_type = 24` and never populates `CompilerOptions.field_313`. Wave 1 confirmed `opts_probe_all_bools.bin` (`f8=1, f10=1, f11=1, f12=1, f17=2`) produced bit-identical `.dgc0` output to `opts_default.bin`.
- Production firmware images under `device production firmware images` (`1.1 925724637`) contain **zero** `huffman` or `codebook` strings.

### 3.2 `CompilerOptions.HuffmanCompressionConfig` (`field_313`)

Reverse-engineered from `TcParseTable @ 0x04eda8d0` and `HuffmanCompressionPass` (`0x2963800`, pass descriptor `0x14c39d`):

| Field | Struct Offset | Type | Default | Semantics |
|---:|---|---|---|---|
| `1` | `+0x50` | `bool` | `false` | `enable_huffman_compression` — top-level pass gate (`0x2963800`). |
| `2` | `+0x4c` | `int32` | `224` (`0xe0`) | `huffman_compression_threshold` — max compressed bytes per 256B block before raw fallback. |
| `3` | `+0x51` | `bool` | `false` | `pad_to_256` — pads every compressed block to 256 B (sets `FlexibleDma bit 500 = 1`) for strided/partial slices. |
| `4` | `+0x52` | `bool` | `false` | Disables the `min_block_bytes = 64` raw-fallback floor (`0x2963f5c`). |
| `6` | `+0x53` | `bool` | `false` | Enables parameter codebook & Huffman DMA lowering (`0x254ca5c`). |
| `8` | `+0x28` | `repeated string` | `[]` | `skip_huffman_compression_for_graphs` — graph name exclusion list. |

### 3.3 Exact Hardware Huffman Codec Specification (`0x3cd3570`)

Implemented bit-for-bit in `offline verification harness: huffman_codec.py`:

1. **Block & Chunk Geometry (`0x3cd3570`):**
   - Input parameter buffer is zero-padded to a multiple of **256 bytes** (`BLOCK_BYTES = 256`).
   - Each 256-byte block is split into **8 independent 32-byte chunks** (`CHUNKS_PER_BLOCK = 8`, `CHUNK_BYTES = 32`).
   - Each 32-byte chunk is encoded **LSB-first** into a byte-aligned bitstream (`0x3cd3c2c..0x3cd3d10`); trailing bits in the last byte of each chunk are zero-padded.
2. **Canonical Codebook Construction (`0x4bdb38c`):**
   - Max code length **`L_max = 16`** bits (Package-Merge length-limited when unconstrained depth exceeds 16).
   - Symbols are bucketed by bit length `L = 1..16` (ascending symbol order per bucket). Within each bucket, codewords are assigned in **descending** order (shifted left each bit-length step), then bit-reversed via `rbit(w0) >> (32 - L)` so LSB-first decoding matches prefix trees directly.
3. **Codebook Wire Format (`0x3cd462c..0x3cd47ec`, `kCodebookSizeBytes = 672 = 0x2a0 B = 5,376 bits`):**
   - `[0..511]` (`512 B`): `256 × uint16_t` LE bit-reversed canonical codewords (`0` if symbol absent).
   - `[512..639]` (`128 B`): `256 × 4-bit` LSB-first `(bit_length - 1) & 0xf` (`0` if absent).
   - `[640..671]` (`32 B`): `256 × 1-bit` LSB-first `symbol_present` bitmap.
4. **Per-Block Metadata Wire Format (`0x3cd41d0..0x3cd4400`, `6 B / 256 B block`):**
   - Eight 6-bit integers `(chunk_byte_len - 1) & 0x3f` packed LSB-first into 48 bits (`6 B` LE).
   - **Raw Fallback Rule (`0x3cd3e9c..0x3cd3ea4`):** If `sum(chunk_lens) >= threshold` (`224`) OR (`field_4 == 0` and `sum(chunk_lens) <= 64`) OR any chunk exceeds `64 B`, the block emits raw uncompressed `256 B` and metadata `[32]*8` -> eight `0x1f` -> **`0x7df7df7df7df`** (`6 B`).
5. **Contiguous `ext[2]` Parameter Packing (`0x254f300..0x254f334`):**
   - `[vec0 (compressed data, 64B-aligned) | vec1 (metadata, 64B-aligned) | vec2 (672B codebook)]`.
   - Verified compression on actual Wave-1 `ext[2]` buffers:
     - `huff_q4like_1x1536x1536_i8` (16 symbols, `H = 3.24 b/sym`, `maxL = 10`): `2,359,296 B -> 1,043,360 B` (**`2.261×`**, `9,216/9,216` blocks compressed).
     - `huff_q4like_1x2048x2048_i8` (16 symbols, `H = 3.24 b/sym`, `maxL = 10`): `4,194,304 B -> 1,854,560 B` (**`2.262×`**, `16,384/16,384` blocks compressed).
     - `huff_gauss_1x1536x1536_i8` (255 symbols, `H = 7.72 b/sym`, `maxL = 11`): `2,359,296 B -> 2,415,264 B` (`0.977×`, `0/9,216` compressed — 100% raw fallback).

### 3.4 Exact `FlexibleDma.Huffman` (`0x0a40`) & `HuffmanDma` (`0x0cc0`) ISA Layout

Disassembled from `0x2be9a2c..0x2bebb20`, `GetFieldBitOffset` (`0x3edfb90..0x3edfd08`), `GetFieldBitWidth` (`0x3edfd40`), `0x2abdb0c..0x2abe210`, and `0x254f300..0x254f368`. On `buenos_a0`, all optional `FlexibleDma` extension flags (`TC.4 field 26`, `ScalarCore fields 113, 301, 335`) are `0`. The compiler **never** emits `opcode 0x33` (`HuffmanDma`, `0x0cc0`) in real codegen (only two hardcoded unit-test callers `@ 0x2c1a794` & `0x2c585b8` with constants `0xf4240` / `0x4c4b40`). Real Huffman DMA uses the **6-flit (`96 B`) extension of `FlexibleDma` (`opcode 0x29 -> 0x0a40`)**:

| Bit Range | Width | DIVE Patch / Field | Value / Semantics on `buenos_a0` |
|---:|---:|---|---|
| `0..36` | 37 b | Header | Standard `FlexibleDma` header (`opcode = 0x0a40`). |
| `37..100` | 64 b | **DIVE field `5`** (`::parameter::dma`) | Main parameter base IOVA. |
| `101..434` | 334 b | Standard `FlexibleDma` body | Identical to 4-flit uncompressed `FlexibleDma`. |
| `435..466` | 32 b | `size_0 - 1` | **UNCOMPRESSED** byte count minus 1 (proven via VLOG `0x10da6f`, `0x2964604`, `0x254f318`). |
| `467..498` | 32 b | `size_1 - 1` | **UNCOMPRESSED** byte count minus 1. |
| **`499`** | 1 b | `huffman_enable` | `FlexDma.Huffman[+0x28]`: `1` enables Huffman decompression, `0` disables. |
| **`500`** | 1 b | `pad_to_256` | `FlexDma.Huffman[+0x2c] != 0`: `0` = packed variable-length blocks, `1` = 256B-padded blocks. |
| **`501..564`** | 64 b | **DIVE field `7`** (`::parameter::compressed_data::dma`) | IOVA of `vec0` (compressed bitstream, `Add(ext2, off_vec0)`). |
| **`565..628`** | 64 b | **DIVE field `8`** (`::parameter::metadata::dma`) | IOVA of `vec1` (`6 B`/block metadata, `Add(ext2, off_vec1)`). |
| **`629..692`** | 64 b | **DIVE field `9`** (`::parameter::codebook::dma`) | IOVA of `vec2` (`672 B` codebook, `Add(ext2, off_vec2)`). |
| `693..695` | 3 b | Common tail | `0b000` (moved from `499..501` in 4-flit form). |
| `696..767` | 72 b | Zero padding | Pads instruction to 6 flits (`96 B`, 16B-aligned). |

### 3.5 Wave-2 Live `rango` Silicon Causal Proof

Nine bare-metal `.dgc0` probes were synthesized via our spec-exact DIVE FlatBuffer rewriter (`offline verification harness: gen_wave2_huffman_probes.py`) and executed live on `rango` (`PM=5`, `MAPS_GOOGLE=0`, `PRIVACY_HITS=0`):

| Probe | Shape | `ext[2]` Bytes | `chunk[4]` ISA Configuration | Live Status | `hw_us` (med) | Live Output vs Raw `field5@37` RNE Sim | Causal Proof Role |
|---|---|---:|---|---|---:|---|---|
| **`h0_ctrl_1x16x16`** | `1x16x16` | `256` | 4-flit `0x0a40`, uncompressed | `ok=True` | `66 us` | **16/16 exact (`max_err=0`)** | Validates DIVE FlatBuffer rewriter live on silicon. |
| **`h0b_ctrl_add256_1x16x16`** | `1x16x16` | `512` | 4-flit `0x0a40`, `DIVE Add(param, +256)` over 256B zero poison | `ok=True` | `54 us` | **16/16 exact** (bit-identical to `h0`) | Proves `DIVE Add` offset patching executes live in runtime. |
| **`h1_6flit_nohuff_1x16x16`** | `1x16x16` | `256` | 6-flit `0x0a40`, `bit499=0` (194 zero bits) | `ok=True` | `48 us` | **16/16 exact** (bit-identical to `h0`) | Proves HIB accepts 6-flit `0x0a40` framing cleanly. |
| **`h2_huff_packed_1x16x16`** | `1x16x16` | `864` | 6-flit `0x0a40`, `bit499=1, bit500=0`, packed Q4-like | `ok=True` | `56 us` | **16/16 exact to `x @ raw_ext2[0:256]`** | Hardware ignored Huffman decode and MAC'd raw compressed bytes! |
| **`h3_huff_pad256_1x16x16`** | `1x16x16` | `992` | 6-flit `0x0a40`, `bit499=1, bit500=1`, 256B-padded | `ok=True` | `42 us` | **16/16 exact to `x @ raw_ext2[0:256]`** | Hardware ignored `pad_to_256` Huffman decode and MAC'd raw bytes! |
| **`h4_huff_rawfb_poison_1x16x16`** | `1x16x16` | `1,248` | 6-flit `0x0a40`, **`field5@37 = 256B zeros`**, `field7@501 = real weights`, `field8@565 = 0x7df7df7df7df`, `field9@629 = 672B cb`, `bit499=1` | `ok=True` | `56 us` | **16/16 exact: `[-112]*16` (`sum(x)*(-128)/8`)** | **DEFINITIVE CAUSAL PROOF:** Even in raw-fallback mode, HIB ignored `bit499=1` and `fields 7..9` and read raw zeros from `field5@37`! |
| **`h5_huff_packed_1x1536x1536`** | `1x1536x1536` | `1,043,360` | 6-flit `0x0a40`, `bit499=1`, compressed `2.36 MB -> 1.04 MB` (`size_0=2,359,296`) | **SMMU FAULT** (`write=0 not mapped`) | — | Clean R52 recovery | **CORROBORATING CAUSAL PROOF:** `FlexibleDma` issued raw `2.36 MB` read from `field5@37` past end of `1.04 MB` DMA-BUF! |
| **`h6_huff_packed_1x2048x2048`** | `1x2048x2048` | `1,854,560` | 6-flit `0x0a40`, `bit499=1`, compressed `4.19 MB -> 1.85 MB` (`size_0=4,194,304`) | **SMMU FAULT** (`write=0 not mapped`) | — | Clean R52 recovery | Same SMMU `write=0 not mapped` fault reading `4.19 MB` past `1.85 MB` DMA-BUF! |
| **`h7_huffdma_0cc0_1x16x16`** | `1x16x16` | `928` | 2-flit `opcode 0x33` (`0x0cc0`) prepended before 4-flit `0x0a40` | `ok=True` | `77 us` | **16/16 exact** (bit-identical to `h0`) | HIB silently skips `opcode 0x0cc0` without faulting or decoding. |

**Conclusion (`[CAUSALLY_PROVEN]`):** Shipping `buenos_a0` HIB silicon / firmware `1.1 925724637` has **no active Huffman decompression engine** on `FlexibleDma` (`0x0a40`) or `HuffmanDma` (`0x0cc0`). Hardware Huffman **cannot** bypass the `27.96 GB/s` INT8 DMA ceiling on Pixel 10.

---

## 4. `SubbyteConstantRemapper`, `W4A8`, `W4A16`, and Composition Analysis

### 4.1 `SubbyteConstantRemapper` (`W4A8`)
- **Compiler RE (`vtable @ 0x4ce9c50`, `Run() @ 0x2513e68`, desc `@ 0x16ff5d`):**  
  Literally documented in-binary as `"This pass replaces subbyte constants with a constant layer with fixed8 types. Currently only handles 4-bit subbyte constants."` (`0x2513ff0: cmp w8, #0xc` -> converts 4-bit constant to `fixed8`).
- **`is_native_subbyte_graph` (`0x1b1ce0`, logger `@ 0x22fc5cc`):** Only activates when the target config exposes native sub-byte VICA unpacking; on `buenos_a0` (`v7_config`), integer quantized `W4A8` graphs hit `SubbyteConstantRemapper` first.
- **Wave-1 Live Proof (`w4a8_pc_1x1536x1536.dgc0`):**
  - Input TFLite had 4-bit packed weights (`1,179,648 B`).
  - Compiled `ext[2]` is **`2,371,584 B`** (`1536 × 1536 × 1 B` + `1536 × 8 B` per-channel params).
  - Inspecting `ext[2][:2,359,296]` confirms **every weight byte is expanded to 8-bit `u8 ∈ [0..15]`** (`len(unique) == 16`).
  - Live `PM=5` latency: **`98 us`** (`ext[2]` bandwidth = `24.23 GB/s`, identical to INT8 `W8A8` `99–111 us`). **0% DRAM bandwidth reduction.**

### 4.2 `W4A16` (`w4a16_deq_i4_1x1536x1536` vs `w8a16_deq_i8_1x1536x1536`)
When 4-bit weights enter via `DEQUANTIZE -> FLOAT FC` (`w4a16_deq_i4_1x1536x1536`), they bypass `SubbyteConstantRemapper` and produce a `1,185,792 B` `ext[2]` (`(1536×1536)/2 + 6,144 B` scales). Exact chunk-by-chunk diff against `w8a16_deq_i8_1x1536x1536.dgc0` proves why this path is broken on `buenos_a0`:
- `chunk[0, 1, 2, 3, 6, 8]` are **100% byte-identical**.
- `chunk[4]` (`FlexibleDma`) halves the weight transfer (`2,359,296 B -> 1,179,648 B`).
- `chunk[5]` (16 × 80-byte `0x980` narrow-to-wide tile DMA packets) changes:
  - `u32[16]` (`+64 B` inside each 80B packet): `0x2020_0000` (`(1<<21) | (128<<22)`, `zp_w = 128`) -> **`0x0220_0000`** (`(1<<21) | (8<<22)`, `zp_w = 8`) — **STILL `bit 21 = 1` (`1:2 u8->FP16` unpack), NOT `1:4` nibble unpack!**
  - `u32[8]` (`+32 B`): `0x3 -> 0x1` (halved beat count).
- Compiler assertion at `0x1c3e71` explicitly checks `config_.tile().support_narrow_to_wide_type_conversion() == config::Tile::INT8_TO_BFLOAT`; **`INT4_TO_BFLOAT` has zero occurrences** in `libedgetpu_tflite_compiler.so`. Live execution was safely skipped per the Wave 1 safety guard because Phase 5 already proved `0x0220_0000` under-delivers half the weights on `buenos_a0`.

### 4.3 Huffman + `0x980` `1:2` (`W4 -> Huffman -> u8 -> FP16`) Composition Verdict
In the compiler architecture, HIB Huffman decompression (`FlexibleDma.Huffman` operating on `ext[2]` -> SRAM) is orthogonal to Tile `0x980` narrow-to-wide `1:2` `u8 -> FP16` conversion (operating on SRAM -> MAC). However, because Wave 2 causally proved `FlexibleDma.Huffman` is inactive on `buenos_a0` silicon, **`W4 -> Huffman -> u8 -> FP16` cannot execute on shipping Pixel 10 hardware**.

---

## 5. Large-Tile `W8A16` (Native `FP16` 10-Bit Mantissa + `FP32` Accumulator)

### 5.1 Major Discovery: Native `FP16` (10-Bit Mantissa) + `FP32` Accumulator — NOT `BF16`!

Although compiler enum names say `INT8_TO_BFLOAT`, live `rango` Wave-1 outputs (`6/6` models, `104,992` total output floats) prove `buenos_a0` actually computes `W8A16` in **IEEE `FP16` (10-bit mantissa) inputs/weights with an `FP32` MAC accumulator**:

1. **Per-Channel `W8A16` (`5/5` models: `1x64x32`, `1x1536x1536`, `1x2048x2048`, `64x1536x1536`, `hybrid_1x1536x1536`):**
   - Computing `acc_live = y_live_f32 / scales_f32` reveals that **100.00% of elements (`103,456 / 103,456`)** have their bottom **13 IEEE-754 mantissa bits (`& 0x1fff`) == `0`** (exact `FP16` 10-bit mantissa), whereas bottom 16 bits (`& 0xffff`, `BF16` 7-bit mantissa) are zero on only ~12.5% (`1/8`, uniform random)!
   - On `w8a16_deq_i8_1x64x32`, `acc_live` matches **`fp16(fp16(x) @ W_int8)`** on **`32/32` elements with `0` ULP error** (`relL2 = 0.000` vs `fp16` sim; `relL2 = 2.801e-4` vs `FP32`, **4.6×–5.7× tighter** than `BF16` `1.28e-3..1.60e-3`!).
   - **Why `z13 = 100%` on per-channel?** The 9-chunk per-channel graph executes two stages: (1) `conv` MAC writes the unscaled dot product to an intermediate SRAM tensor in **`FP16`**, (2) VPU `mul` multiplies that `FP16` tensor by `scales_f32` and writes `FP32` output.
   - **CRITICAL OVERFLOW HAZARD (Per-Channel 9-Chunk):** Because the unscaled intermediate SRAM tensor is `FP16` (max finite value **`65,504`**), any channel whose unscaled dot product `|x @ W_i8| > 65,504` (common in LLM `down_proj` with `K=8960` and activation outliers `|x| ~ 20–50`) will overflow to `±Inf` **before** per-channel scaling!
2. **Per-Tensor `W8A16` (`w8a16_deq_i8_pt_1x1536x1536`, 7 chunks):**
   - The scalar `scale` folds directly into the MAC epilogue (no intermediate `FP16` SRAM tensor, no VPU `mul` stage, `0` scales DMA).
   - `y_live` matches **`fp32(fp16(x) @ W_int8) * scale`** with **`relL2 = 2.760e-7`** (median **`2` FP32 ULPs**, `z13 = 0/1536`), and achieves **`relL2 = 1.877e-4`** vs pure `FP32` (limited solely by `fp32 -> fp16` conversion of input `x`!).
   - **Zero intermediate `FP16` overflow hazard**, faster execution (`138 us` vs `142 us`), and simpler 7-chunk DIVE topology!
3. **`DEQUANTIZE+FC` vs `HYBRID_FC`:** `w8a16_hybrid_i8_1x1536x1536.dgc0` and `w8a16_deq_i8_1x1536x1536.dgc0` are **100% byte-identical across all 9 instruction chunks** (`chunk[0..8]`).
4. **DWS0 I/O Buffer Format Note:** `dgc0_dump.py` labels DWS0 `data_type = 7` as `FLOAT16`, but actual DMA-BUF I/O sizes (`in_bytes = 4*M*K`, `out_bytes = 4*M*N`) and live outputs prove DWS0 `data_type = 7` is **`FLOAT32`** — the TPU converts `FP32` input -> `FP16` on-chip during DMA/VPU ingest and writes `FP32` output directly!

### 5.2 Wave-1 Live `rango` `PM=5` Benchmark Table (`#202`)

| Model (`sgs = 1` for all) | Shape `(M,K,N)` | Chunks | `ext[2]` Bytes | `hw_us` med (min) | Wall (ms) | Total DMA GB/s | GOPS | `relL2` vs FP32 | `relL2` vs BF16 | `relL2` vs FP16/Acc Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `huff_q4like_1x1536x1536_i8` *(W8A8 ctrl)* | `1×1536×1536` | 7 | `2,359,296` | `99` (`97`) | `0.430` | `23.86` | `47.66` | *(INT8 err ≤ 1)* | — | — |
| `huff_gauss_1x1536x1536_i8` *(W8A8 ctrl)* | `1×1536×1536` | 7 | `2,359,296` | `111` (`100`) | `0.388` | `21.28` | `42.51` | *(INT8 err = 0)* | — | — |
| `huff_q4like_1x2048x2048_i8` *(W8A8 ctrl)* | `1×2048×2048` | 7 | `4,194,304` | `168` (`163`) | `0.459` | `24.99` | `49.93` | *(INT8 err ≤ 1)* | — | — |
| `huff_gauss_1x2048x2048_i8` *(W8A8 ctrl)* | `1×2048×2048` | 7 | `4,194,304` | `163` (`161`) | `0.425` | `25.76` | `51.46` | *(INT8 err = 0)* | — | — |
| `w4a8_pc_1x1536x1536` *(remapped 4b->8b)* | `1×1536×1536` | 9 | `2,371,584` | `98` (`95`) | `0.432` | `24.23` | `48.15` | *(remapped u8)* | — | — |
| `w8a16_deq_i8_1x64x32` | `1×64×32` | 9 | `2,176` | `23` (`21`) | `0.325` | `0.11` | `0.18` | `2.801e-4` | `1.281e-3` | **`0.000e+00`** (`32/32` 0-ULP) |
| **`w8a16_deq_i8_pt_1x1536x1536`** *(PT)* | `1×1536×1536` | **7** | `2,359,296` | **`138`** (`137`) | `0.547` | **`17.19`** | `34.19` | **`1.877e-4`** | `1.541e-3` | **`2.760e-7`** *(FP32 acc!)* |
| `w8a16_deq_i8_1x1536x1536` *(PC)* | `1×1536×1536` | 9 | `2,365,440` | `142` (`141`) | `0.407` | `16.74` | `33.23` | `2.798e-4` | `1.601e-3` | `1.779e-5` *(FP16 mid)* |
| `w8a16_hybrid_i8_1x1536x1536` *(PC)* | `1×1536×1536` | 9 | `2,365,440` | `142` (`140`) | `0.470` | `16.74` | `33.23` | `2.671e-4` | `1.540e-3` | `1.357e-5` *(FP16 mid)* |
| `w8a16_deq_i8_1x2048x2048` *(PC)* | `1×2048×2048` | 9 | `4,202,496` | **`233`** (`232`) | `0.584` | **`18.11`** | `36.00` | `2.803e-4` | `1.419e-3` | `1.231e-7` *(FP16 mid)* |
| **`w8a16_deq_i8_64x1536x1536`** *(Prefill!)* | **`64×1536×1536`** | 9 | `2,365,440` | **`589`** (`587`) | `0.965` | `5.35` | **`512.72`** | **`2.778e-4`** | `1.511e-3` | `1.043e-5` *(FP16 mid)* |

*Note:* `Total DMA GB/s = (ext[2] + in_bytes + out_bytes) / hw_us` (where `in_bytes = M*K` and `out_bytes = M*N` for INT8 controls, or `4*M*K` and `4*M*N` for FP32-I/O W8A16). For `64×1536×1536`, total DMA is `(2,365,440 + 393,216 + 393,216) / 589 us = 5.35 GB/s` (parameter-only `ext[2] / hw_us = 4.02 GB/s`).

### 5.3 Universal `W8A16` `ext[2]` Weight Packing Contract (`[CAUSALLY_PROVEN]`)

Verified **100% byte-exact** across all 6 compiled `W8A16` `.dgc0`s (`M ∈ {1, 64}`, `K ∈ {64, 1536, 2048}`, `N ∈ {32, 1536, 2048}`, both per-channel and per-tensor):

- **Preconditions:** `N % 32 == 0`, `K % 2 == 0`, symmetric `INT8` weights `W[n, k] ∈ [-128, 127]` (`zp_w = 0`).
- **Weight Tensor Layout (`ext[2][0 .. N*K - 1]`):**
  ```
  Shape:  [N / 32, K / 2, 32, 2]   (row-major C-order, 64 bytes per inner [32 × 2] block)
  Offset: (((n >> 5) * (K >> 1) + (k >> 1)) * 32 + (n & 31)) * 2 + (k & 1)
  Byte:   uint8_t(W[n, k]) ^ 0x80  (maps signed int8 [-128..127] -> unsigned [0..255], matching 0x980 zp=128)
  ```
- **Per-Channel Scales Tail (9-chunk PC only; OMITTED in 7-chunk PT):**
  - Immediately appended at `ext[2][N*K .. N*K + 4*N - 1]`: raw little-endian `float32 scales[N]`.
- **Hardware Instruction Signature (`0x980` narrow-to-wide tile DMA):**
  - 5-flit (`80 B` = `20 × uint32_t`) packet, `u32[16]` (`+64 B`) = **`0x2020_0000`** (`(1 << 21) | (128 << 22)` -> `INT8_TO_FP16` `1:2` unpack with `zp_w = 128`).

---

## 6. C++17 Integration Blueprint for Milestone 8B

### 6.1 Recommended Architecture for `libopenbuenos` Milestone 8B

1. **Do NOT implement Huffman or `W4A16` on `buenos_a0`:** Both are causally ruled out on shipping Pixel 10 silicon.
2. **Keep Overlapped Multi-Threaded NEON `W8A8` as the Default Decode (`M=1`) Engine:** At `1x2048x2048`, `W8A8` achieves `163–168 us` (`25.0–25.8 GB/s` single-tile, `27.96 GB/s` ring-pipelined) vs `W8A16`'s `233 us` (`18.07 GB/s`, ~`1.40×` slower).
3. **Add Optional 7-Chunk Per-Tensor `W8A16(FP16)` (`scale = 1.0`) + NEON Per-Channel Scale Epilogue** for two high-value workloads:
   - **Outlier-Heavy Layers (`down_proj`, `lm_head`, Layer 0–1):** Eliminates **100% of CPU activation quantization, `uniq_k` outlier detection, outlier-column cache (`OPENBUENOS_MAX_OUTLIER_COLS`), and outlier FMA peeling** while upgrading precision from `~3e-3` INT8 to **`1.88e-4`** (`FP16` inputs + `FP32` accumulator) — with **zero `FP16` intermediate overflow risk** because the 7-chunk PT graph outputs `FP32` directly!
   - **High-Accuracy `M=64` Prefill (`pp512`):** Single `64x1536x1536` tile runs at **`589 us` (`512.7 GOPS`, `relL2 = 2.78e-4`)** with raw `FP32` I/O — zero CPU quantization overhead, zero `M=64` saturation retries (`sat_retries = 0` by construction!).

### 6.2 Drop-In C++17 / ARM64 NEON Weight Packer (`[N/32, K/2, 32, 2]`)

Packs row-major `int8_t W_nk[N][K]` directly into a zero-copy DMA-BUF (`ext[2]`) with no intermediate allocation (`N % 32 == 0`, `K % 16 == 0`):

```cpp
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

// Pack W_nk (row-major [N, K] int8) -> buenos_a0 W8A16 ext[2] layout [N/32, K/2, 32, 2] of (w ^ 0x80).
// Parallelizable trivially across n_blk = 0 .. (N/32 - 1) on ob_worker_pool!
void pack_w8a16_ext2_nblk_neon(
    const int8_t* __restrict__ w_nk,
    uint8_t* __restrict__ dst_ext2,
    size_t K,
    size_t n_blk_begin,
    size_t n_blk_end) noexcept {
    const uint8x16_t kXor80 = vdupq_n_u8(0x80);
    const size_t k_pairs = K >> 1;
    const size_t blk_stride_bytes = k_pairs * 64;  // [K/2, 32, 2] bytes per 32-row N-block

    for (size_t nb = n_blk_begin; nb < n_blk_end; ++nb) {
        uint8_t* blk_dst = dst_ext2 + nb * blk_stride_bytes;
        const size_t n_base = nb * 32;

        // Process 16 K-columns (= 8 K-pairs) per row, writing 2 bytes into 8 consecutive 64B slots
        for (size_t n_in = 0; n_in < 32; ++n_in) {
            const uint8_t* src = reinterpret_cast<const uint8_t*>(w_nk + (n_base + n_in) * K);
            uint8_t* dst = blk_dst + n_in * 2;
            size_t k = 0;
            for (; k + 16 <= K; k += 16) {
                uint8x16_t v = veorq_u8(vld1q_u8(src + k), kXor80);
                uint16x8_t pairs = vreinterpretq_u16_u8(v);
                // Store 8 x 16-bit K-pairs at stride 64 bytes (next K-pair's [32,2] slice)
                *reinterpret_cast<uint16_t*>(dst + 0 * 64) = vgetq_lane_u16(pairs, 0);
                *reinterpret_cast<uint16_t*>(dst + 1 * 64) = vgetq_lane_u16(pairs, 1);
                *reinterpret_cast<uint16_t*>(dst + 2 * 64) = vgetq_lane_u16(pairs, 2);
                *reinterpret_cast<uint16_t*>(dst + 3 * 64) = vgetq_lane_u16(pairs, 3);
                *reinterpret_cast<uint16_t*>(dst + 4 * 64) = vgetq_lane_u16(pairs, 4);
                *reinterpret_cast<uint16_t*>(dst + 5 * 64) = vgetq_lane_u16(pairs, 5);
                *reinterpret_cast<uint16_t*>(dst + 6 * 64) = vgetq_lane_u16(pairs, 6);
                *reinterpret_cast<uint16_t*>(dst + 7 * 64) = vgetq_lane_u16(pairs, 7);
                dst += 8 * 64;
            }
            for (; k < K; k += 2) {
                dst[0] = src[k] ^ 0x80;
                dst[1] = src[k + 1] ^ 0x80;
                dst += 64;
            }
        }
    }
}
```

### 6.3 7-Chunk Per-Tensor `W8A16` + Overlapped NEON Scale/Accumulate Epilogue

For `K`-tiled GEMM (`K = sum(K_tile)`), run each tile with the **7-chunk per-tensor `W8A16` graph (`scalar_scale = 1.0f`, `ext[2] = N_tile * K_tile` bytes)** and fuse per-channel weight scales `scales_n[N_tile]` + `K`-tile accumulation in the existing overlapped NEON drain loop:

```cpp
// Fused NEON epilogue for 7-chunk Per-Tensor W8A16 (scalar_scale = 1.0f):
//   k_tile_idx == 0: out[m, n]  = raw_f32[m, n] * scales_n[n]
//   k_tile_idx  > 0: out[m, n] += raw_f32[m, n] * scales_n[n]
// Eliminates intermediate FP16 overflow (>65504), saves 4*N scales DMA, and drops 2 chunks (138us vs 142us).
void w8a16_pt_epilogue_neon(
    const float* __restrict__ raw_tpu_f32,   // [M, N_tile] FP32 from 7-chunk W8A16 tile
    const float* __restrict__ scales_n,      // [N_tile] per-channel (or block-group) FP32 weight scales
    float* __restrict__ out_f32,             // [M, N_total] destination (row stride ld_out)
    size_t M,
    size_t N_tile,
    size_t ld_out,
    bool store_first) noexcept {
    for (size_t m = 0; m < M; ++m) {
        const float* src = raw_tpu_f32 + m * N_tile;
        float* dst = out_f32 + m * ld_out;
        size_t n = 0;
        if (store_first) {
            for (; n + 16 <= N_tile; n += 16) {
                vst1q_f32(dst + n +  0, vmulq_f32(vld1q_f32(src + n +  0), vld1q_f32(scales_n + n +  0)));
                vst1q_f32(dst + n +  4, vmulq_f32(vld1q_f32(src + n +  4), vld1q_f32(scales_n + n +  4)));
                vst1q_f32(dst + n +  8, vmulq_f32(vld1q_f32(src + n +  8), vld1q_f32(scales_n + n +  8)));
                vst1q_f32(dst + n + 12, vmulq_f32(vld1q_f32(src + n + 12), vld1q_f32(scales_n + n + 12)));
            }
            for (; n < N_tile; ++n) dst[n] = src[n] * scales_n[n];
        } else {
            for (; n + 16 <= N_tile; n += 16) {
                vst1q_f32(dst + n +  0, vfmaq_f32(vld1q_f32(dst + n +  0), vld1q_f32(src + n +  0), vld1q_f32(scales_n + n +  0)));
                vst1q_f32(dst + n +  4, vfmaq_f32(vld1q_f32(dst + n +  4), vld1q_f32(src + n +  4), vld1q_f32(scales_n + n +  4)));
                vst1q_f32(dst + n +  8, vfmaq_f32(vld1q_f32(dst + n +  8), vld1q_f32(src + n +  8), vld1q_f32(scales_n + n +  8)));
                vst1q_f32(dst + n + 12, vfmaq_f32(vld1q_f32(dst + n + 12), vld1q_f32(src + n + 12), vld1q_f32(scales_n + n + 12)));
            }
            for (; n < N_tile; ++n) dst[n] += src[n] * scales_n[n];
        }
    }
}
```

### 6.4 Integration Guardrails & Next Steps for Milestone 8B

1. **Input Dynamic Range Check (`FP16`):** Hardware converts input `FP32 -> FP16` (`max = 65,504`, `min_normal = 6.1e-5`). Clamp or prescale any activation row with `max(|x|) > 32768.0f` by a power-of-two `row_scale` (folded into `scales_n` on CPU) — zero extra cost and prevents `FP16` input saturation.
2. **Block-Quantized `Q8_0`/`Q4_0` Scale Granularity:** For `ggml` `Q8_0` (`block_size = 32` along `K`), `W8A16` requires a single scale per `(n, K_tile)` (or per-channel across `K_tile`). Rescale each `Q8_0` block inside a `K_tile` onto a common per-channel `s_max[n, kt]` (identical to existing `pack_ext2` per-channel re-quantization for `W8A8`, with `Q8_0` blocks within a `K_tile` already sharing tight dynamic ranges), OR group `K_tile = 256..512` to minimize re-quantization loss while keeping `ext[2]` bandwidth high.
3. **Clean-Room `W8A16` Tile Emitter:** `w8a16_deq_i8_pt_1x1536x1536.dgc0` (7 chunks, `16,224 B` `ext[0]`) and `w8a16_deq_i8_64x1536x1536.dgc0` (`589 us` / `512.7 GOPS`) serve as exact reference oracles for adding a clean-room `ob_w8a16_emitter` alongside `ob_gemm_emitter` in `libopenbuenos` (no `libedgetpu_tflite_compiler.so` runtime dependency, keeping `NEEDED = [libdl.so, libm.so, libc.so]`).

---

## 7. Artifact & Reproduction Index

- **Offline Verifier (`ALL CHECKS PASSED`):** `offline verification harness: verify_phase8_huffman_w4a16.py`
- **Extracted `TargetConfig` Protobufs (14 configs):** `offline verification harness: configs/`
- **Bit-Exact Hardware Huffman Codec:** `offline verification harness: huffman_codec.py`
- **Wave-1 Compiler-Oracle Generator & Inputs:** `offline verification harness: gen_wave1_oracle.py`, `offline verification harness: wave1_tflite/`
- **Wave-2 Bare-Metal Huffman Probe Generator & DIVE Rewriter:** `offline verification harness: gen_wave2_huffman_probes.py`, `offline verification harness: wave2_huffman_probes/`
- **Wave-1 Live `rango` (`PM=5`) Telemetry & Compiled `.dgc0`s:** `live device suite: wave1_oracle/summary.json`
- **Wave-2 Live `rango` (`PM=5`) Causal Proof Telemetry & Outputs:** `live device suite: wave2_huffman/summary.json`
