# Phase 3 `isa-dma`: `buenos_a0` DMA, TTU & SRAM Tiling

Offline clean-room recovery and machine verification of the Google Tensor G5 (`laguna` / `buenos_a0`) DMA descriptors (`0xa40`, `0x9c0`), narrow/wide SRAM TTU loop setups (`0x940`, `0x980`), DIVE VM IR `PutBits` runtime address bindings, and SRAM/loop tiling regimes.

- **Full Bitfield Specification**: [dma_ttu_bitfields.md](dma_ttu_bitfields.md)
- **Automated Cross-Batch Verifier**: [verify_dma_decoder.py](../../tools/verify_dma_decoder.py)
- **Raw Machine Verification Report**: `verify_dma_report.json`

---

## Executive Summary

1. **Flexible DMA (`0xa40`, 64 B)**: Transfer byte counts span **32 bits** across `w6:w7` (`prod_dma_bytes - 1` at `[466:435]`, `cons_dma_bytes - 1` at `[498:467]`), fixing the 13-bit (`0x1FFF`) display truncation in [dgc0_dump.py](../../tools/dgc0/dgc0_dump.py#L628-L629). Inputs/parameters transfer at exact byte lengths; outputs align UP to 4 bytes (`((out_bytes + 3) // 4) * 4`). Runtime DRAM pointers are patched via DIVE `PutBits(offset=37, bit_width=64)` (`[359:296]`).
2. **Output DMA Tail (`0x9c0`, 64 B at `chunk[-1][64:128]`)**: Encodes `single_beat_le32` (`w8[50]`), 32-byte beat count minus 1 (`w9 = ((ceil32(out_dma)//32)-1) << 5`), `multi_beat_gt32` (`w12[45]`), tail/head beat bounds (`w12[54:50] = (out_dma-1)%32`, `w12[59:55] = min(out_dma,32)-1`), and `w14 = (out_dma << 15) | 0x4040`.
3. **SRAM Ring & TTU Descriptors (`0x940` 48 B & `0x980` 80 B)**: `0x940` packs 4 x 31-bit TTU dims (`dims[1] = 0x7f008000 | max(0, ceil(B/16384)-1)`), `16 KiB` (`in`) / `64 KiB` (`par`) ring counters/slack in `w3`, `sram_off_mod32k_32b` in `w0[47:33]`, and `(M << 2) + sram_off_32b` in `w4[25:16]`. `0x980` shifts that payload by `+1` bit (`bit 65+`) for quarter-tiles/half-banks and carries absolute `sram_off // 64` in `w5[63:42]` plus the 16-tile multicast mask `[519:504]`.
4. **DMA Lowering & Tiling Regimes**: `Reshape` compiles to RV64 `Copy` (`0` TPU chunks); aligned `Slice` (`origin`, `c_off4`, `c_off16`, `w_off3`) compiles directly in `5` chunks via TTU; unaligned `Slice` (`b2_slice_c_off1`) lowers to a `7`-chunk parameterized `identity_conv` (`params = 96 B`). `>= 64 KiB` elementwise ops switch from unrolled `0x900` instructions (`ch[4] = 41,952 B` at `32 KiB`) to a fixed hardware loop (`ch[3] = 6,176 B`, `ch[4] = 3,616 B`).

---

## Live `rango` Sweep Coverage

| Batch | Models | Result on `rango` (`the on-device runner`) | Focus |
|---|---:|---|---|
| **Baseline** | 3 | Pre-compiled (`fc`, `add`, `conv2d`) | Anchors `chunk[0..6]` layout |
| **Batch 0** (`phase3_batch0`) | 27 | `27/27` pass | `dma_reshape_*`, `dma_slice_*`, `dma_concat_*`, GEMM/vec/pipe2 |
| **Batch 1** (`phase3_isa_dma_b1`) | 35 | `35/35` compile/run/verify | `4 B .. 262,144 B`, `[C=4,16,32,64]`, `Add` (`32, 64, 32K`), `FC` (`M=1..8`), `Slice`, `Concat` (`axis=1,2`) |
| **Batch 2** (`phase3_isa_dma_b2`) | 12 | `12/12` compile/run/verify (`err<=1` quantize, `err=0` rest) | `1, 5, 17, 49, 65, 4128, 98304 B`, `FC 3x16x16` & `1x256x64` (`par=16640 B`), `concat 4+8`, `slice_c_off1`, `slice_w_off3` |
| **Total Verified** | **77** | **`74` TPU + `3` RV64 `Copy`** | **0 serial/SoC ID leaks** |

---

## Verified Decoder Output

Command:
```bash
python3 tools/verify_dma_decoder.py
```

Observed stats (`exit code 0`):
```text
Total .dgc0 files discovered: 77
Summary stats: {
  "models_total": 77,
  "models_tpu": 74,
  "models_rv64_only": 3,
  "cnt_chunk1": 74,
  "cnt_0xa40": 184,
  "cnt_0xa40_in": 89,
  "cnt_0xa40_par": 21,
  "cnt_0xa40_out": 74,
  "cnt_0x9c0": 74,
  "cnt_0x940": 110,
  "cnt_0x980": 338
}
RV64-only (0 TPU chunks, 3): ['batch0/dma_reshape_1x16x16x16_to_1x256x16', 'batch0/dma_reshape_1x256_to_1x16x16', 'batch0/dma_reshape_1x64x1x64_to_1x16x4x64']
Failures (0):
Odd 0xa40 (0):
Odd 0x9c0 (0):
Odd 0x940 (0):
```

### Representative Multi-Regime Differential Signatures

| Model | Port Bytes | `0xa40` `prod-1` (in / out) | `0x9c0` `w8[50]` | `0x9c0` `w9` | `0x9c0` `w12` | `0x9c0` `w14` | `SCR` / `field_14` |
|---|---:|---|---:|---|---|---|---:|
| `b2_q_1x1x1x1` | `1 -> 1` | `0` (`1`) / `3` (`4`) | `1` | `0x0` | `0x018c000000000000` | `0x24040` | `192,512` / `2,368` |
| `b2_q_1x1x1x17` | `17 -> 17` | `16` (`17`) / `19` (`20`) | `1` | `0x0` | `0x09cc000000000000` | `0xa4040` | `192,512` / `2,624` |
| `b2_q_1x1x1x49` | `49 -> 49` | `48` (`49`) / `51` (`52`) | `0` | `0x20` | `0x0fcc200000000000` | `0x1a4040` | `192,512` / `2,816` |
| `b2_q_1x1x1x65` | `65 -> 65` | `64` (`65`) / `67` (`68`) | `0` | `0x40` | `0x0f8c200000000000` | `0x224040` | `192,512` / `2,880` |
| `b2_q_1x129x1x32` | `4,128 -> 4,128` | `4,127` / `4,127` | `0` | `0x1000` | `0x0ffc200000000000` | `0x8104040` | `200,704` / `34,624` |
| `b1_q_1x1024x1x64` | `65,536 -> 65,536` | `65,535` / `65,535` | `0` | `0xffe0` | `0x0ffc200000000000` | `0x80004040` | `323,584` / `525,120` |
| `b2_q_1x1536x1x64` | `98,304 -> 98,304` | `98,303` / `98,303` | `0` | `0x17fe0` | `0x0ffc200000000000` | `0xc0004040` | `389,120` / `787,264` |
| `b1_q_1x4096x1x64` | `262,144 -> 262,144` | `262,143` / `262,143` | `0` | `0x3ffe0` | `0x0ffc200000000000` | `0x200004040` | `716,800` / `2,097,984` |
| `b2_fc_3x16x16` | `48` (`par=320`) | `47` / `319` / `47` | `0` | `0x20` | `0x0fbc200000000000` | `0x184040` | `192,512` / `3,072` |
| `b2_slice_c_off1` | `1,024 -> 512` (`par=96`) | `1,023` / `95` / `511` | `0` | `0x1e0` | `0x0ffc200000000000` | `0x1004040` | `192,512` / `3,392` |

---

## Repository Artifact Index (`isa-dma`)

### Documentation
- [dma_ttu_bitfields.md](dma_ttu_bitfields.md) — Exact bitfield tables, compiler packer mapping, SRAM scaling, and `[M, K, N]` patching recipe.
- [README.md](README.md) — This summary and verification report.

### Scripts & Analysis Work Files
- [verify_dma_decoder.py](../../tools/verify_dma_decoder.py) — Standalone cross-batch DMA/TTU verifier (`77` `.dgc0`s).
- `gen_dma_sweep.py` & `sweep_batch1/manifest.json` — Batch 1 sweep (`35` models).
- `gen_dma_sweep_b2.py` & `sweep_batch2/manifest.json` — Batch 2 sweep (`12` models).
- `analyze_batch1.py` & `batch1_analysis.txt` — Batch 1 differential dump.
- `inspect_batch0.py` & `batch0_summary.txt` — Batch 0 summary.
