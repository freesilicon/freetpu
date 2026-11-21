# Tensor G5 (`buenos_a0`) DMA, TTU & SRAM Tiling Bitfields

Exact offline-recovered `buenos_a0` DMA, Tensor Traversal Unit (TTU), DIVE IR binding, and SRAM scaling specification across **77 compiled `.dgc0` artifacts** (`3` baseline + `27` Batch 0 + `35` Batch 1 + `12` Batch 2, spanning `1 B .. 262,144 B`) cross-checked against `libedgetpu_tflite_compiler.so` (`lga_tpu_20260603.0.0`, CL `925724637`).

- **Machine-verified decoder**: [verify_dma_decoder.py](../../tools/verify_dma_decoder.py) (`0` failures across all `77` `.dgc0`s: `74` TPU models + `3` RV64-only `Reshape` models)
- **Hardware config source**: `compiler__buenos_a0_config.txt` (`2` cores, `16` tiles `[4x4 snake]`, `786,432 B` narrow memory/core, `1,048,576 B` shared memory unit, `1` instruction queue, `1` DMA descriptor queue, `1` DMA data queue, `4` sync watchers)

---

## 1. DIVE VM IR -> Hardware Chunk DMA Lifecycle

All `buenos_a0` `.dgc0` binaries carry an RV64 DIVE interpreter (`tensor[3]`) and a `DIVE` module (`tensor[2]`, `"dive_vm_ir"`). Hardware execution follows strict chunk staging:

### Chunk Topologies Observed

| Topology | Chunk Count | Models Observed | Chunk Sequence |
|---|---:|---|---|
| **RV64 `Copy` only (`Reshape`)** | `0` | All 3 `dma_reshape_*` | Pure RV64 DIVE `Copy` op (`output_0`); `hardware_instruction_chunks = []`. |
| **Unary / aligned `Slice`** | `5` | `Quantize`, `Relu`, `Logistic`, aligned `Slice` (`origin`, `4_4`, `h_contig`, `c_off4`, `c_off16`, `w_off3`) | `ch[0]` prolog (`320 B`) -> `ch[1]` `before_flexible_dma__tpu_input` (`80 B`) -> `ch[2]` `after_request_flexible_dma` input (`64 B`) -> `ch[3]` `before_flexible_dma__tpu_output` -> `ch[4]` `after_request_flexible_dma` output + `finish_and_sleep`. |
| **Dual-input (`Add`, `Mul`, `Concat`) / Parameterized (`FC`, `Conv2D`, unaligned `Slice`)** | `7` | `Add`, `Mul`, `Concat` (`axis=1,2,3`), `FC`, `Conv2D`, `b2_slice_c_off1` | `ch[0]` prolog -> `ch[1..2]` `input_0` -> `ch[3..4]` `input_1` or `parameter` -> `ch[5..6]` `output_0` + `finish_and_sleep`. |
| **2-Op Graph (`Conv1x1 + Add`)** | `9` | `pipe2_conv1x1_add_1x16x16x16` | Prolog + `in0` (`1,2`) + `param` (`3,4`) + `in1` (`5,6`) + `out` (`7,8`). Note: `pipe2_add_logistic` and `pipe2_fc_relu` fuse compute into `7` chunks. |

### Exact DIVE Host <-> TPU Sequence (FC Example)

1. Host caches cleaned/invalidated via `CacheCleanInvalidate` (`op_type=61`), base pointers retrieved via `GetTensorAddress` (`op_type=59`).
2. `DispatchHardwareInstruction` (`op_type=47`) sends `chunk[0]` (`program_prolog_hw_instructions_0`, always `320 B`).
3. **For each DMA stream** (`input_0`, optional `input_1`/`parameter`, `output_0`):
   - `DispatchHardwareInstruction` sends **`before_flexible_dma_*`**: prepares narrow/wide SRAM ring + TTU traversal (`0x940`, `0x980`). On `input_0`, `chunk[1]` is always `80 B` (`0x680` prolog guard + `48 B` `0x940`).
   - **`PutBits` (`op_type=50`)**: patches `chunk_index = k`, **`offset = 37` bytes (`bit 296`), `bit_width = 64`** inside the next chunk's `0xa40` descriptor with the runtime DRAM tensor address.
   - `DispatchHardwareInstruction` sends **`after_request_flexible_dma_*`** (`chunk[k]`): issues the `64 B` **`0xa40`** flexible DMA request. For `output_0`, the final chunk also appends the `64 B` **`0x9c0`** output DMA tail immediately at `[64:128]` followed by narrow-to-wide unpack / sync / `finish_and_sleep`.
4. `WaitForFenceCompletion` (`op_type=51`, `"halt"`) + final `CacheCleanInvalidate`.

> **Important `dgc0_dump.py` caveat**: [dgc0_dump.py](../../tools/dgc0/dgc0_dump.py#L628-L629) applies a 13-bit mask `0x1FFF` to `w6 >> 51` when printing `decoded_dma_head_64b`, which truncates transfers `> 8,192 B`. The true `0xa40` size fields span **32 bits** across `w6:w7`.

---

## 2. Flexible DMA 64-Byte Descriptor (`opcode = 0xa40`)

Emitted at `chunk[2]`, `chunk[4]`, ..., and `chunk[-1][0:64]`. Verified across **184 / 184 instances** (`89` inputs, `21` params, `74` outputs) with `0` failures. Compiler packer: `FUN_02ce69fc` (`compiler_bitpackers.c:2884-3150`, calls `FUN_03fdeacc` with 32-bit fields for producer/consumer transfer sizes and `FUN_03fdf088` for header/constant slots).

| Global Bits | `[word:bits]` | Width | Field | Verified Value / Formula |
|---|---|---:|---|---|
| `[11:0]` | `w0[11:0]` | 12 | `opcode` | `0xa40` (`[5:0]=0`, `[11:6]=0x29`) |
| `[27:12]` | `w0[27:12]` | 16 | `hw_seq_id` | Monotonic per-instruction sequence counter (`9` on `ch[2]`, increments through program) |
| `[31:28]` | `w0[31:28]` | 4 | `reserved_0` | Always `0` |
| `[39:32]` | `w0[39:32]` | 8 | `dma_channel_tag` | `(slot << 5) \| kind`: `kind` `1`=input, `2`=parameter, `3`=output (`0x01`, `0x21`, `0x41`, `0x22`, `0x23`, `0x43`, `0x63`) |
| `[63:40]` | `w0[63:40]` | 24 | `reserved_1` | Always `0` |
| `[100:64]` | `w1[36:0]` | 37 | `reserved_2` | Always `0` |
| `[101]` | `w1[37]` | 1 | `flex_dma_enable` | Always `1` (`w1 = 0x00000020_00000000`) |
| `[295:102]` | `w1[63:38], w2, w3, w4[39:0]` | 194 | `reserved_3` | Always `0` |
| **`[359:296]`** | `w4[63:40] \| (w5[39:0] << 24)` | **64** | **`dram_base_addr`** | `0` at rest in `.dgc0`; **byte offset `37`, `bit_width = 64`** patched by DIVE `PutBits` at runtime |
| `[360]` | `w5[40]` | 1 | `reserved_4` | Always `0` |
| `[361]` | `w5[41]` | 1 | `mode_flag_b361` | Always `1` (`w5[63:40] = 0x2`) |
| `[430:362]` | `w5[63:42], w6[46:0]` | 69 | `reserved_5` | Always `0` |
| `[434:431]` | `w6[50:47]` | 4 | `slot_b431` | Always `0x1` (`w6[50:0] = 0x00008000_00000000`, `FUN_02ce69fc:3022`) |
| **`[466:435]`** | `((w6 >> 51) \| ((w7 & 0x7FFFF) << 13)) & 0xFFFFFFFF` | **32** | **`prod_dma_bytes - 1`** | Total AXI/DMA transfer byte count minus 1 (`FUN_03fdeacc` 32-bit pack) |
| **`[498:467]`** | `(w7 >> 19) & 0xFFFFFFFF` | **32** | **`cons_dma_bytes - 1`** | Equals `prod_dma_bytes - 1` on all `184` verified descriptors |
| `[511:499]` | `w7[63:51]` | 13 | `reserved_6` | Always `0` |

### Exact Transfer Size Alignment Rule

- **Inputs (`kind == 1`) & Parameters (`kind == 2`)**: `dma_bytes = exact_tensor_bytes` (no alignment padding; verified down to `1 B`, `5 B`, `17 B`, `49 B`, `65 B`, `96 B`, `4128 B`, `16,640 B`, `262,144 B`).
- **Outputs (`kind == 3`)**: **rounded UP to 4-byte multiple**:
  ```python
  out_dma_bytes = ((out_bytes + 3) // 4) * 4
  ```
  Verified on unaligned `Quantize` sweeps: `1->4`, `5->8`, `17->20`, `49->52`, `65->68`, and exact multiples (`4..262144`).

---

## 3. Output DMA Tail 64-Byte Descriptor (`opcode = 0x9c0`)

Immediately follows the output `0xa40` at `chunk[-1][64:128]` (words `w8..w15`). Verified across **74 / 74 outputs** (`4 B .. 262,144 B` aligned DMA bytes) with `0` failures.

| Global Bits (from `0x9c0` base) | `[rel_word:bits]` | Width | Field | Verified Value / Formula |
|---|---|---:|---|---|
| `[11:0]` | `w8[11:0]` | 12 | `opcode` | `0x9c0` (`[5:0]=0`, `[11:6]=0x27`) |
| `[27:12]` | `w8[27:12]` | 16 | `hw_seq_id` | Always `output_0xa40.hw_seq_id + 1` |
| `[49:28]` | `w8[49:28]` | 22 | `reserved_0` | Always `0` |
| **`[50]`** | `w8[50]` | 1 | **`single_beat_le32`** | `1` iff `out_dma_bytes <= 32` (`ceil(out_dma_bytes / 32) == 1`), else `0` |
| `[63:51]` | `w8[63:51]` | 13 | `reserved_1` | Always `0` |
| **`[127:64]`** | `w9[63:0]` | 64 | **`beat32_count_m1_shl5`** | `(((out_dma_bytes + 31) // 32) - 1) << 5` (`0` for `4..32 B`, `0x20` for `52/64 B`, `0x40` for `68 B`, `0x17fe0` for `98,304 B`, `0x3ffe0` for `262,144 B`) |
| `[255:128]` | `w10, w11` | 128 | `reserved_2` | Always `0` |
| `[300:256]` | `w12[44:0]` | 45 | `reserved_3` | Always `0` |
| **`[301]`** | `w12[45]` | 1 | **`multi_beat_gt32`** | `1` iff `out_dma_bytes > 32`, else `0` (`1 ^ w8[50]`) |
| `[303:302]` | `w12[47:46]` | 2 | `reserved_4` | Always `0` |
| `[307:304]` | `w12[51:48]` | 4 | `const_0xc` | Always `0xC` |
| **`[310:306]`** *(overlaps top bits of `0xC` via `[51:50]`)* | `w12[54:50]` | 5 | **`tail_beat_bytes_m1`** | `(out_dma_bytes - 1) % 32` (`3` for `4 B` / `68 B`, `19` for `20 B` / `52 B`, `31` for `32 B` / `4128 B` / `>=32 B` multiples; bit `1` of `0xC` at `[51:50]=0b11` encodes `(rem % 4 == 3)`!) |
| **`[315:311]`** | `w12[59:55]` | 5 | **`first_beat_cap_m1`** | `min(out_dma_bytes, 32) - 1` (`3` for `4 B`, `19` for `20 B`, `31` for `>=32 B`) |
| `[319:316]` | `w12[63:60]` | 4 | `reserved_5` | Always `0` |
| `[383:320]` | `w13` | 64 | `reserved_6` | Always `0` |
| **`[447:384]`** | `w14[63:0]` | 64 | **`dma_byte_len_tag`** | **`(out_dma_bytes << 15) \| 0x4040`** (`4 B -> 0x24040`, `20 B -> 0xa4040`, `52 B -> 0x1a4040`, `68 B -> 0x224040`, `98304 B -> 0xc0004040`, `262144 B -> 0x200004040`) |
| `[511:448]` | `w15` | 64 | `reserved_7` | Always `0` |

*Note on `w12[51:48]` vs `w12[54:50]`*: Because `out_dma_bytes` is always 4-byte aligned, `(out_dma_bytes - 1) & 3 == 3`, so `w12[51:50] == 0b11` and `w12[49:48] == 0b00` (`w12[51:48] == 0xC`) holds unconditionally across all `74` outputs.

---

## 4. `chunk[1]` (`before_flexible_dma__tpu_input_from_input_0`, `80 B`)

Verified identically across **74 / 74** TPU models (`10 x u64`):

- **`w0..w3` (`bytes [0:32]`)**: Fixed `32 B` prolog/fence guard (`opcode = 0x680` + `0x3c0` sync):
  - `w0 = 0x00000000_7FFFF680`
  - `w1 = 0x000FFFF8_000003C0`
  - `w2 = 0x00000000_00000000`
  - `w3 = 0x00000000_00000000`
- **`w4..w9` (`bytes [32:80]`)**: Standard **48-byte `0x940`** descriptor configuring `input_0` (`hw_seq_id = 0x0008`).

---

## 5. Narrow SRAM Ring & TTU Loop Setup (`opcode = 0x940`, `48 B` / `6 x u64`)

Appears in `chunk[1][32:80]` (`input_0`), at tails of `before_flexible_dma__tpu_input1` (`Add`/`Mul`/`Concat`), and at heads of `before_flexible_dma__tpu_output` for parameterized kernels (`FC`, `Conv2D`, `b2_slice_c_off1`). Verified across **110 / 110 instances** (`89` inputs, `21` params) with `0` failures.

| Global Bits | `[word:bits]` | Width | Field | Input (`w3[31:0] = 0x10010800`) | Parameter (`w3[31:0] = 0x80010800`) |
|---|---|---:|---|---|---|
| `[11:0]` | `w0[11:0]` | 12 | `opcode` | `0x940` (`[5:0]=0`, `[11:6]=0x25`) | `0x940` |
| `[27:12]` | `w0[27:12]` | 16 | `hw_seq_id` | `0x0008` on `ch[1]`; higher on `in1` | Monotonic seq ID |
| `[31:28]` | `w0[31:28]` | 4 | `reserved_0` | `0` | `0` |
| `[32]` | `w0[32]` | 1 | `is_input_flag` | `1` | `0` |
| **`[47:33]`** | `w0[47:33]` | 15 | **`sram_off_mod32k_32b`** | `(sram_byte_off % 32768) // 32` (`0` for `in0`; non-zero on `in1` of `Add`/`Concat` `<16 KiB`: `32B->1`, `64B->2`, `256B->8`, `4096B->0x80`; wraps to `0` at `32768B`) | `0` |
| `[48]` | `w0[48]` | 1 | `enable_bit48` | `1` (`w0[63:48] = 1`) | `1` |
| `[94:64]` | `comb[30:0]` | 31 | **`dim0`** | `0` if `B<=32`; else `0x7F0081FF` (`511` -> `512*32B = 16 KiB` inner span) | `0x78008FFF` (`4095` -> `64 KiB` param span) |
| `[125:95]` | `comb[61:31]` | 31 | **`dim1`** | `0` if `B<=32`; else `0x7F008000 \| max(0, ceil(B/16384)-1)` (`0` `<16K`, `1` `32K`, `3` `64K`, `5` `96K`, `7` `128K`, `15` `256K`) | `0x78008000` (even for `16,640 B`) |
| `[156:126]` | `comb[92:62]` | 31 | **`dim2`** | `0` if `B<=32`; else `0x7F008000` | `0x78008000` |
| `[187:157]` | `comb[123:93]` | 31 | **`dim3`** | `0` if `B<=32`; else `0x7F008000` | `0x78008000` |
| `[191:188]` | `w2[63:60]` | 4 | `reserved_1` | `0` | `0` |
| `[223:192]` | `w3[31:0]` | 32 | `stream_kind_cfg` | `0x10010800` | `0x80010800` |
| **`[238:227]`** | `w3[46:35]` | 12 | **`ring_slots_m1`** | Let `B32 = ceil32(B)`: `0` if `B<=32 or B>=16384`, else `(B32 // 32) - 1` | `(par32 // 32) - 1` (`par32 = ceil32(par_bytes)`) |
| `[242]` | `w3[50]` | 1 | `ring_active` | `0` if `B<=32 or B>=16384`, else `1` | `1` |
| **`[255:244]`** | `w3[63:52]` | 12 | **`ring_slack_16b`** | `0` if `B<=32 or B>=16384`, else **`0x400 - (B32 // 16)`** (`16 KiB` ring budget) | **`0x1000 - (par32 // 16)`** (`64 KiB` param ring budget) |
| `[271:256]` | `w4[15:0]` | 16 | `port_mode_lo16` | `0x0210` | `0x0311` |
| **`[281:272]`** | `w4[25:16]` | 10 | **`row_or_sram32_tag`** | **`(M << 2) + sram_off_32b`** (`0x004` default; FC `M=1,2,3,4,8 -> 0x04,0x08,0x0c,0x10,0x20`; `Add`/`Concat` `in1` adds `sram_off_mod32k_32b`: `0x05,0x06,0x0c,0x24,0x44,0x84`) | `0x004` |
| `[297:282]` | `w4[41:26]` | 16 | `bank_ring_base` | `0xFFFF` (`0x3fffc...`) normally; **`0x8000`** (`0x20000...`) on `b1_add_1x512` `in1` at `32 KiB` SRAM boundary | `0xFFF8` (`0x3ffe0...`) |
| `[319:298]` | `w4[63:42]` | 22 | `reserved_2` | `0` | `0` |
| `[383:320]` | `w5[63:0]` | 64 | `tail_const` | Always `0x00000000_00000004` | Always `0x00000000_00000004` |

*(Where `comb = w1 \| (w2 << 64)`.)*

---

## 6. Per-Tile / Quarter-Ring TTU Setup (`opcode = 0x980`, `80 B` / `10 x u64`)

Inside `before_flexible_dma` chunks (`ch[3]`, `ch[5]`), the compiler emits `80 B` **`0x980`** TTU descriptors with `w0[63:49] == 1` (`338` verified instances across `74` TPU models; distinguish from non-DMA `0x980` conv3x3 compute ops with `w0[63:49] = 0x1000` and `stride = 26 x u64`):

- **Bit-shift equivalence with `0x940`**: In `0x980`, `w1[0]` (`global bit 64`) is `0`, so the `4 x 31`-bit `dims` and `w3` payload start at **`bit 65`** (`(w1 \| (w2 << 64)) >> 1` and `(w3 \| (w4 << 64)) >> 1`), decoding via the exact same `0x940` formulas for that **tile slice** (`quarter_bytes = total_bytes / 4`, or `par_bytes / 2` when `N=64` FC splits across dual banks)!
- **Key addressing & multicast fields**:
  - `w0[32]`: `1` (input/narrow) vs `0` (param).
  - `w0[48:33]`: **`(sram_byte_off % 32768) // 16`** (`16 B`-aligned offset modulo `32 KiB`, e.g. `add 4K`: `0x000, 0x040, 0x080, 0x0c0`; `b2_q 96K`: `0x000, 0x600, 0x400, 0x200`).
  - `w0[63:49]`: `0x0001` (`bit 49 = 1`).
  - **`w5[63:42]`**: **Absolute narrow SRAM byte offset divided by 64** (`sram_byte_off // 64` -> `w5` increases by `quarter_bytes << 36`! Verified on `b1_add_1x512` where `in1` starts at `32,768 B -> w5[63:42] = 0x200..0x380` even while `w0[48:33]` wraps mod `32 KiB`).
  - **`w7[63:56] \| (w8[7:0] << 8)` (`[519:504]`)**: **16-tile multicast bitmask** (`0x000f, 0x00f0, 0x0f00, 0xf000` for 4x4 clusters in `<=32 KiB`; unicast `0x0001, 0x0080, 0x0100, 0x8000` at corner tiles in `>=64 KiB` loop schedules; `0xffff` on all-tile broadcast). `w9 == 0`.

---

## 7. DMA Ops (`Reshape`, `Slice`, `Concat`) & `>= 64 KiB` Regime Shift

1. **`RESHAPE`**: Zero TPU instructions (`chunks = 0`). Lowers purely to RV64 DIVE `Copy`.
2. **`SLICE`**:
   - **Aligned (`begin[3] % 4 == 0`, `size[3]` lane-friendly)**: Compiles in **`5` chunks** using TTU `0x980` (`b1_slice_origin`, `b1_slice_c_off4`, `b1_slice_c_off16`, `b1_slice_h_contig`, `b2_slice_w_off3`, and Batch 0 slices). No parameter tensor is emitted.
   - **Unaligned channel offset (`b2_slice_c_off1`, `begin=[0,0,0,1], size=[1,8,8,8]`)**: Lowers to a **`7`-chunk parameterized kernel** (`output_0.padding.identity_conv::parameter`, **`params = 96 B`**), with `ch[3..4]` DMA'ing the packed `96 B` filter/pad table!
3. **`CONCAT` (`axis=1, 2, 3`)**: Compiles in **`7` chunks** regardless of axis or unaligned channels (`4 + 8 = 12` in `b2_concat_c_4_8`); `in1` shares the narrow SRAM buffer via `sram_off_32b` when total inputs fit in `< 32 KiB`, or rolls to the `0x8000` second bank (`32 KiB` boundary).
4. **`>= 64 KiB` Hardware Loop Regime Shift** (`QUANTIZE` sweep `4..262,144 B`):
   - `<= 32 KiB`: Computes via unrolled `0x900` instructions (`ch[4]` grows linearly: `2,272 B` at `1K`, `6,112 B` at `4K`, `21,472 B` at `16K`, **`41,952 B` at `32K`**).
   - **`>= 64 KiB` (`64K`, `96K`, `128K`, `256K`)**: Compiler switches to a **fixed-size hardware loop schedule** (`ch[3] = 6,176 B`, `ch[4] = 3,616 B`, `.dgc0 = 660,096 B` identical across `64K..256K`!). Inside `ch[4]`, `16` per-tile `0x400` slots (`stride = 14 x u64 = 112 B`) scale ONLY at:
     - `w17 + 14*k`: `0x2_0000_0000 | ((dma_bytes // 512) << 20)` (`128, 192, 256, 512` per-tile 32B beats)
     - `w23 + 14*k`: `[8:0] = (passes_of_32K - 1)` (`1, 2, 3, 7`), `[22:9] = (dma_bytes // 16) - 1` (`0xfff, 0x17ff, 0x1fff, 0x3fff`).

### Host Scratchpad (`dive_scratch_<N>`) & `Executable.scalars.field_14`

- **`dive_scratch` allocation** (4 KiB pages):
  - `184,320 B` (`180 KiB`) base for RV64-only small `Reshape` (`256 B`).
  - `192,512 B` (`188 KiB`) floor for all TPU ops `<= 1,024 B` (quantize/slice/concat small) and **every** `FC` model (even `par = 16,640 B`).
  - Scales cleanly by `4,096 B` pages above the floor: for `Quantize` (`4K..256K`): `192,512 + 2 * ceil_4k(N_bytes)` (`4K -> 200,704`, `16K -> 225,280`, `32K -> 258,048`, `64K -> 323,584`, `96K -> 389,120`, `128K -> 454,656`, `256K -> 716,800`). For `Add`: `192,512 + 3 * N_bytes` at `4K` (`204,800`) and `32K` (`290,816`).
- **`field_14`** (cycles / cost scalar in `Executable.scalars`): for `Quantize >= 4,096 B`, exact `8 * N_elems + 832` (`4096->33600`, `32768->262976`, `65536->525120`, `98304->787264`, `131072->1049408`, `262144->2097984`).

---

## 8. `[M, K, N]` FC Patching Recipe (for `the GEMM ISA spec`)

To patch an existing 7-chunk INT8 `FULLY_CONNECTED` template (`in = M*K`, `out = M*N`, `par = packed_bytes`, `out_dma = ((M*N+3)//4)*4`):

1. **`chunk[1]` (`80 B`, `[32:80]` `0x940` input_0)**:
   - Update `dims[0..3]` (`0` if `M*K<=32`, else `[0x7f0081ff, 0x7f008000 | d1, 0x7f008000, 0x7f008000]`).
   - Update `w3_hi32` (`0` if `M*K<=32 or M*K>=16384`, else `((0x400 - ceil32(M*K)//16)<<20) | (1<<18) | (((ceil32(M*K)//32)-1)<<3)`).
   - **Update `w4[25:16] = M << 2`** (`w4 = 0x000003ff_fc000210 | ((M << 2) << 16)`).
2. **`chunk[2]` (`64 B` `0xa40` input_0)**: Keep `PutBits(ch=2, off=37, w=64)` target zeroed; set `[466:435]` & `[498:467]` = `M*K - 1`.
3. **`chunk[3]` (`before_flexible_dma__parameter`)**: Notice `ch[3]` emits `M` row-wise `0x980` setups (and `K`-tile splits above `K > 32`); coordinate with `the GEMM ISA spec` matmul instruction patching inside `ch[3]` and `ch[5]`!
4. **`chunk[4]` (`64 B` `0xa40` param)**: Set `[466:435]` & `[498:467]` = `par_bytes - 1`. Keep `[359:296]` zeroed.
5. **`chunk[5][0:48]` (`0x940` param)**: Set `w3_hi32 = ((0x1000 - ceil32(par_bytes)//16)<<20) | (1<<18) | (((ceil32(par_bytes)//32)-1)<<3)` (`par_bytes <= 64 KiB`). At `chunk[5][48:128]` (`0x980` param), note `N=64` emits dual half-bank `0x980` descriptors (`par_bytes // 2` per bank when `> 8 KiB` / wide `N`).
6. **`chunk[6]` (`output_0`)**: Patch `0xa40` (`[0:64]`) `[466:435]` & `[498:467]` = `out_dma - 1`, and **`0x9c0` (`[64:128]`)** using `w8[50]`, `w9`, `w12`, `w14` formulas in §3, plus update DGC0 FlatBuffer port `size_bytes_u32/u64`, `executable_hash` / `model_hash` if checked, and `dive_scratch_<N>` (`192,512` holds for all tested FC sizes up to `1x256x64`).

---

## 9. Observed vs Inferred vs Unknowns Ledger

| Claim | Classification | Evidence Scope |
|---|---|---|
| `0xa40` 32-bit `prod-1`/`cons-1` at `[466:435]`/`[498:467]`, `tag` `[39:32]`, `PutBits` off=`37` len=`64`, `out` 4-byte ceiling alignment | **Observed** | Verified `184/184` across all `74` TPU `.dgc0` models (`1 B..262,144 B`) + `FUN_02ce69fc` (`compiler_bitpackers.c`) |
| `0x9c0` `w8[50]`, `w9`, `w12` (`[45]`, `[54:50]`, `[59:55]`), `w14 = (out_dma<<15)\|0x4040`, `w10=w11=w13=w15=0` | **Observed** | Verified `74/74` output tails (`4..262,144 B` DMA sizes) |
| `0x940` `4x31`-bit `dims`, `w3` `16K`/`64K` ring counters/slack, `w4[25:16] = (M<<2) + off32`, `chunk[1]` `0x680` header | **Observed** | Verified `110/110` `0x940` packets (`89` in, `21` par) |
| `0x980` `+1` bit shift vs `0x940`, `w0[48:33]` mod-32K `//16`, `w5[63:42]` abs `//64`, `w7:w8` `[519:504]` 16-tile mask | **Observed** | Verified `338/338` DMA `0x980` packets across `74` TPU models |
| `Reshape` RV64-only (`0` chunks), unaligned `Slice` lowering to `96 B` `identity_conv`, `>=64 KiB` loop shift (`w17`, `w23`) | **Observed** | Verified by live `rango` compile+execute+verify (`27+35+12`) + diff inspection |
| Hardware semantics of constant flags `w1[37]`, `w5[41]`, `w6[47]` (`0xa40`), `w4[41:26] = 0xFFFF/0xFFF8/0x8000` (`0x940`), and `w5[40:0]` `0x8400` vs `0x8c00` (`0x980` at exact `64 KiB`) | **Inferred** | Consistent across all sweeps (`0x8000` appears specifically at `32 KiB` bank wrap in `b1_add_1x512`; `0x8400` appears solely at exact single-pass `16 KiB/tile` = `64 KiB`), not perturbed standalone via `dgc0_patcher` live run |
| Exact `0xa40` behavior for `prod_dma_bytes != cons_dma_bytes`, async multi-core split (`core_or_cluster_count=2` with DMA `seq` spanning >256 KiB SRAM spill), and `> 256 KiB` / `> 768 KiB` narrow memory spill | **Unknown** | All `77` models stayed `<= 716,800 B` scratchpad (`<= 256 KiB` tensor) and had `prod_dma_bytes == cons_dma_bytes` |
