# Phase 4 Multi-Tile INT8 GEMM ISA Specification (`buenos_a0` / Tensor G5)

Complete clean-room closed-form specification and hardware proof for arbitrary `(M, K, N)` multi-tile INT8 GEMM (`Y = clamp(RNE((X - zp_in) @ W^T + B) * (s_in * s_w / s_out) + zp_out, qmin, qmax)`) on Pixel 10 / Tensor G5 (`buenos_a0`).

---

## 1. End-to-End Verification Summary

| Verification Suite | Scope | Result | Artifacts |
|---|---|---|---|
| `verify_full_isa_81.py` | All **81** compiled INT8 GEMM `.dgc0` models (`phase3_gemm_sweep1` 22, `phase3_batch0` 5, `phase4_gemm_sweep` 42, `phase4_edge_sweep` 12) | **81/81 FULL BIT-EXACT** (`c0=0 c1=0 c2=0 c3=0 c4=0 c5=0 c6=0 ext2=0`, only masked field: 4-bit cyclic tag `c6[-5].lo[15:12]`) | [`verify_full_isa_81.py`](../../tools/verify_full_isa_81.py), [`multitile_synth.py`](../../tools/multitile_synth.py) |
| `dgc0_builder.py` | Dynamic `DIVE` (`ext[0]`) + `DWS0` + `DGC0` container synthesis across all 81 models | **81/81 BIT-EXACT `ext[0]`**, **81/81 `field_14`**, **81/81 `dive_scratch`**, **81/81 `dgc0_dump` round-trip** | [`dgc0_builder.py`](../../tools/dgc0_builder.py) |
| `emit_custom_gemm.py --self-test` | 15 legacy Phase 3 cases + all 81 multi-tile corpus models (zero manual bit overrides) | **15/15 Legacy PASS**, **81/81 Multi-Tile PASS** (`exit 0`) | [`emit_custom_gemm.py`](../../tools/emit_custom_gemm.py) |
| Live `rango` Dual-Runner Proof (`#75`) | **7** novel multi-tile custom `.dgc0`s (NO TFLite compiler, incl. unseen `(10, 32, 32)`) on **both** `tpu_direct_runner --run-dgc0` and zero-Google-`.so` `tpu_baremetal_runner` (`--iters 5`) | **7/7 PASS**, **3,368/3,368 bytes exact (`max_abs_err=0`)**, `bm_vs_direct_byte_identical=7/7`, `maps_has_google_tpu_libs=false` (`7/7`) | `summary.json` |

### Owned Implementation Artifacts (SHA-256)
- [`work/isa-gemm/multitile_synth.py`](../../tools/multitile_synth.py): `sha256: 4db114e83fe310fcbb88dab0a14e91a0d64d0180c2be232140853a9bf2b4728f`
- [`work/isa-gemm/verify_full_isa_81.py`](../../tools/verify_full_isa_81.py): `sha256: 0ff6cada0ea2c12ba5cb6a86a05aa44b81344b42636c81d78043c473c3d4bfa3`
- [`work/isa-gemm/dgc0_builder.py`](../../tools/dgc0_builder.py): `sha256: f020f0f3d878e023d77b0ccbce306753762daddec3b0c6a0fbed05fb422acdbf`
- [`work/isa-gemm/emit_custom_gemm.py`](../../tools/emit_custom_gemm.py): `sha256: 0764ffde6652c1ffd692de51528dc0ccdada54109c952efc8ac88a60148bc66e`

---

## 2. Architecture & Closed-Form Synthesis Equations

### 2.1 Global Packet Sequencing (`pid`) & 4x4 Physical Tile Topology
- Every VLIW packet across `chunk[0..6]` carries a global monotonic 16-bit sequence ID `pid` stamped into flit `0` low word:
  - `0x680`, `0x6c0`: `lo0[43:28] = pid`
  - `0x900`, `0x940`, `0x980`, `0x9c0`, `0xa40`: `lo0[27:12] = pid`
  - All other opcodes (`0x040, 0x200, 0x3c0, 0x400, 0x440, 0x480, 0x4c0, 0x500, 0x5c0, 0x640`): `lo0[63:48] = pid`
  - Chunk ranges: `c0: 0..6` (`7` pkts), `c1: 7..8` (`2` pkts), `c2: 9` (`1` pkt), `c3: 10 .. 10+len(pk3)-1`, `c4: 10+len(pk3)` (`f0_hi_word=0x22`), `c5: c4_pid+1 ..`, `c6: c5_end ..` (`f0_hi_word=0x43`, final `OP_900` unstamped except 4-bit cyclic tag).
- The 16 tiles are addressed logically (`0..15`) in tile-mask fields (`lo0[27:12]`) and mapped to physical mesh coordinates via serpentine (snake) order in `OP_400` destination bitmasks (`f6.hi[53:34]`) and `OP_980` DMA tile masks (`f3.hi[63:56] | f4.lo[7:0]`):
  ```text
  SNAKE_16 = (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12)
  ```

### 2.2 Parameter Buffer Packing (`ext[2]`)
Let `k4 = pad4(K)`, `n4 = pad4(N)`, `n_blk = 32 if N > 32 else max(8, ceil8(N))`, `n_groups = ceil(N / 32)`:
- For each group `g in 0 .. n_groups-1` (`n0 = 32*g`):
  1. If `has_bias`: `n_blk` little-endian `int32` biases (`bias[n0+ni]` if `n0+ni < N` else `0`).
  2. For `kg in 0 .. k4//4 - 1`, `ni in 0 .. n_blk - 1`, `kk in 0 .. 3`:
     - `(W[n0+ni][4*kg+kk] + 128) & 0xFF` if `n0+ni < N and 4*kg+kk < K`
     - `0x80` if `n0+ni < n4 and 4*kg+kk < k4` (zero-value INT8 padding inside active 4-element lanes)
     - `0x00` otherwise (unused alignment channels).

### 2.3 Static & DMA Chunks (`chunk[0]`, `chunk[1]`, `chunk[2]`, `chunk[4]`)
- **`chunk[0]`**: 20 constant flits (`320 B`, 4-quadrant init + barrier).
- **`chunk[1]`**: 5 flits (`80 B`): `OP_680(3c0)` + `OP_940` input DMA descriptor with `fl[4].lo = 0x000003FFFC000210 | (M << 18)`; when `in_chunks = ceil(M*K / 32) > 1`, `c = (in_chunks - 1) * 8` populates `fl[2].hi = 0x3F8040007F0081FF`, `fl[3].lo = 0x0FE010001FC02000`, `fl[3].hi = ((((0x3FE4 - 4*c) & 0xFFFF) << 16 | c) << 32) | 0x10010800`.
- **`chunk[2]`**, **`chunk[4]`**, **`chunk[6][0..3]`**: 4-flit `OP_a40` host/param DMA packets with `dma_enc = (n_bytes - 1) * 8`, `f3 = (dma_enc << 80) | (dma_enc << 48) | (1 << 47)`, `f0_hi_word in {0x01, 0x22, 0x43}` (`c2`: `M*K`, `c4`: `len(ext2)`, `c6`: `M*pad4(N)`). The RV64 DIVE interpreter patches the 64-bit IOVA into bit offset `37` of each via `PutBits`.

### 2.4 Grand Unified Shadow-CSR Architecture (`HDR_DATA` -> `VICA_CSR` -> `CSR_040`)
A key Phase 4 discovery unifying all 81 models with **zero special-case hacks**:
1. **`HDR_DATA` (`OP_200`, 20 flits)** starting at **bit 77** (`f0.hi` bit 13) is literally **98 contiguous 25-bit hardware CSR registers `0x00 .. 0x61`**:
   ```text
   reg[r] = (hdr_bits >> (77 + 25 * r)) & 0x1FFFFFF    for r in 0x00..0x61
   ```
2. **`VICA_CSR` (`OP_200` compressed)** and **`CSR_040` (`OP_040`)** share an identical CSR-delta packet encoding:
   - `u32_entry(r, val) = ((val & 0x1FFFFFF) << 7) | (r & 0x7F)` in ascending register ID order.
   - `hi0 = (u32s[0] << 32) | ((2*n_csrs + 1) << 12) | (0x50 if ef_flag else 0)`, remaining `u32s[1:]` packed 4 per flit.
3. **Single Shadow-State Diffing**:
   - In `chunk[3]`, `HDR_DATA(M, K, s_in, zp_in, sw_unit)` initializes shadow state `H`.
   - If `K > 32 and K % 32 != 0`, `VICA_CSR` updates the trailing partial tile (`mask = 1 << (K // 32)`) to `H_vica` for `r in 0x00..0x59` (preserving `0x57` for `q > 1`, `0x18` for `q > 1, d_rem < 16`, and `0x07, 0x16` for `q == 1, d_rem < 16`).
   - `CSR_040[0]` (`mask = (1 << max(1, N // 32)) - 1`) emits register `r in 0x00..0x61` **iff** `target0[r] != H_vica[r]`.
   - If `N > 32 and N % 32 != 0`, `CSR_040[1]` (`mask = 1 << (N // 32)`) emits `r` **iff** `target1[r] != target0[r]`.
4. **Closed-Form Resolution of Phase 3 Quirks (`g01` `0x5e=2` and `g08` `0x5c[21]=1`)**:
   - **`g01` (`0x5e = 2`)**: `HDR_DATA` writes `fp32((-128 - zp_in) * s_in)` at `b17 + 259`; for `zp_in = 17, s_in = 0.03125`, `(-145)*0.03125 = -4.53125` sets `bf16` LSB = 1 at bit `77 + 25*0x5e + 24`, making `H[0x5e] = 0x1000002 != target[0x5e] = 2`, so shadow-diffing emits `0x5e = 2` automatically.
   - **`g08` (`0x5c[21] = 1`)**: The upper clamp at `b17 + 222` (spanning `0x5c[24:21]`, `0x5d[24:0]`, `0x5e[2:0]`) is computed in FP32 as `u_max = fp32_bits(f32(f32((qmax - zp_out) * s_out_f) * f32(1.0 / s_out_f)))`. For `s_out = 0.03`, `f32(f32(127 * 0.03) * f32(1 / 0.03)) = 0x42FE0001` (`u_max[0] = 1 -> 0x5c[21] = 1`).
   - **`0x57` and `0x61[18]`**: `0x57` carries `((zp_in + 128) & 0xFF) << 4` (unchanged from `HDR_DATA`); `0x61[18]` is `has_bias` (`bias_enable`).

### 2.5 Compute Streams (`Stream A`, `Stream H`, `Stream B`, `Stream C`) & `OP_6c0`
Let `k_u = pad4(K) // 4`, `k_inner = min(12, k_u)`, `k_passes = ceil(k_u / 12)`, `k_rem12 = k_u % 12`, `m_ge_48 = (M >= 48)`, `m_eff = M // 4 if m_ge_48 else M`, `n4_t = n_tile // 4`:
- **Weight streaming**: `stream_w = (K > 64 and pad4(K) * n_pad_w >= 4096)`, `sw_unit = next_pow2(last_980) // 0x800`.
- **Stream A (`0x06..0x12`)**:
  - `k_passes == 1`: `[(k_u-1, 1), (0, -(k_u-1))]` for `M=1`; `[(M*k_u-1, -(M-1)*k_u+1), (0, -(M*k_u-1))]` for `2 <= M < 48`; `[(k_u-1, 1), (m_eff-1, -(m_eff*k_u-1))]` for `M >= 48`.
  - `k_passes >= 2, k_rem12 == 0`: `[(11, 1), (k_passes-1, -(k_u-1))]`.
  - `k_passes >= 2, k_rem12 != 0` (unified for all `M >= 1`): `[(11, k_u-11), (m_eff-1, -(M-1)*k_u+1), (k_passes-1, -((M-1)*k_u + 12*k_passes - 1))]`.
- **Stream H (`0x17..0x24`)**: `hh = 0x2000 | sw_unit` (zeros) for `M = 1`; `hh = sw_unit`, `[(n4_t, m_eff-1), (-n4_t*(m_eff-1), 0)]` with `0x24 |= 0x10000` for `M >= 2`.
- **Stream B (`0x2a..0x36`)**: `b_st = -4*(k_inner - 1)`, bounds `[k_inner-1, (m_eff-1 if M>=2 else k_passes-1), (k_passes-1 if M>=2 and k_passes>=2 else 0)]`.
- **Stream C (`0x3c..0x42`)**: for `M >= 2`, `0x3c = ((m_eff-1)<<12) | (0 if m_ge_48 else 2)`, `0x3d..0x42 = (-2*(M-1)) & 0xFFF` (if `not m_ge_48`), with `0x3d |= (k_passes-1)<<12` when `k_passes >= 2`; for `M = 1, k_passes >= 2`, `0x3c = (k_passes-1)<<12`.
- **Even-`M` `OP_6c0` lane pointers**: factored by `q, ef = q_factor_m(M)` (`m_grp = M // q`):
  - `m_grp == 4`: 8 packets (`g = 0..3`, `addr = base + g * (M//4) * d_sub`, `hi = (1<<g)<<56 | (addr<<16) | {0x8000, 0x8400}`). For `M >= 48`, `chunk[3]` emits `OP_6c0` twice (in `mid3` at `base=0` and before `CSR_040` with `out_off = M*min(32, k4)`).
  - `m_grp == 2`: 4 packets (`lane 0 @ base`, `lane 1 @ base + (M//2)*d_sub`), plus 2 extra packets on the partial trailing tile mask (`1 << (d_pad4 // 32)`) at `base + (M//2)*(d_pad4 % 32)` when `d_pad4 > 32 and d_pad4 % 32 != 0`.

### 2.6 Universal 2D DMA Rewind & Multi-Tile Output Scatter/Gather (`chunk[5]` + `chunk[6]`)
- **Universal 2D DMA Rewind Flit 1** (shared across `pre3`, `post3`, Param SRAM `OP_440`, `c5` scatter `440/480/400`):
  For total rewind stride `step = inner * (y - 1)` bytes:
  ```text
  f1.lo = (((-step * 8) & 0xFFFF) << 48) | ((0x80000000 - step * 0x400) & 0xFFFFFFFF)
  f1.hi = (0x0002000000000000 - (step << 28)) | 0xFF
  ```
- **Output Tile Partition (`compute_output_tiles(M, N)`)**:
  - `M == 1`: `ceil(N / 32)` tiles, `tid = g`, `len = min(32, pad4(N) - 32*g)`.
  - `2 <= M <= 16`: `row_stride = 16 // M`, `T_row = min(pad4(N)//4, row_stride)` balanced multiples of 4B per row; logical tile ID `tid = row * row_stride + j` (leaving unused row-slot gaps when `pad4(N)//4 < 16 // M`, e.g., `g12_m2`: `0..3, 8..11`; `g15_m3`: `0..3, 5..8, 10..13`).
  - `M > 16`: 16 tiles (`tid = 0..15`), each `(M * pad4(N)) // 16` bytes (`2D` scatter when `nb not in {4, 8, 16, 32}`).
- **Per-Tile Channel-1 Sync Token & `chunk[6]` Coalescing**:
  - Each active tile at sequence index `seq_idx` with logical ID `tid` receives channel-1 token:
    ```text
    tok = c3_440_ch1 + c5_ch1 + seq_idx + (c3_44c if tid < c3_dst_cnt else 0)
    ```
  - Tiles sharing `(tok, nb)` coalesce into one `OP_400` packet in `chunk[6]` with `ch_hi = ((4*tok + 1) << 12) | 0x44C`.
  - `b5` emits `n5 = len(tile_specs)` `OP_900` sync packets with `w2 = (b5_0 + 2*k) << 8`, `b5_0 = c3_440_ch1 + c5_ch1 + 1`, and tail `c1_tot = b5_0 + 2*n5 - 1`, `c2_tot = c3_ch2 + c5_ch2`.

### 2.7 Dynamic `DGC0` / `DWS0` / `DIVE` Container Synthesis (`dgc0_builder.py`)
- **`ext[0]` (`DIVE` FlatBuffer)**: 160 B root/header + `chunk[6]..chunk[0]` laid out in reverse order at 32-byte aligned offsets with 12 B wrapper tables pointing to a shared 6 B vtable (`06 00 08 00 04 00`) at `c0_tbl - 6` + 2976 B `functions[0]` suffix with int64 `output_0=[M,N]` (`+2664,+2672`), `input_0=[M,K]` (`+2848,+2856`), and `PutBits` VA patch ops (`chunk[2]@37`, `chunk[4]@37`, `chunk[6]@37`, 64-bit).
- **`DWS0`**: patches `input_0=[M,K]`, `output_0=[M,N]`, `ext[0]`/`ext[2]` sizes, `dive_scratch_<size>` (`196608` if `M * pad4(N) >= 1024` else `192512`), `MultiExecutable` IO/param sizes, `field_14` (tile SRAM watermark), unique 64-bit `executable_hash`, and unique 19-digit `Token_<id>` (avoiding `libbuenos` cache collisions).
- Preserves `DGD0` (4096 B aligned), `DWS0` (64 B aligned), `ext[1]` RV64 interpreter ELF (`638977 B`), `0x0a` gap padding, and 64 B ExternalBuffer/file alignment.

---

## 3. Live Dual-Runner Silicon Proof on `rango` (7 Novel Multi-Tile Custom Microkernels)

All 7 kernels were synthesized strictly offline via [`emit_custom_gemm.py --emit-multitile`](../../tools/emit_custom_gemm.py) with **zero TFLite compiler involvement**, custom weights & non-zero biases, non-power-of-two `s_out`, asymmetric zero-points, and `fp32_vs_tflite_max_abs_diff == 0`, then executed on live `rango` (`--iters 5`) via **both** `/data/local/tmp/tpu_direct_runner --run-dgc0` and zero-Google-`.so` `/data/local/tmp/tpu_open_stack/tpu_baremetal_runner` (`phase4_custom_multitile_live/summary.json`):

| Kernel | `(M, K, N)` | Quant & Act | Features Exercised | `dgc0` SHA-256 (prefix) | Exact Bytes | `max_abs_err` | Direct `hw_us` (`mean_ms`) | Baremetal `hw_us` (`mean_ms`) | `bm == direct` & Zero Google `.so` |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `custom_4x32x32_nonpot005_relu` | `(4, 32, 32)` | `s_out=0.05`, `zp=-11/+19`, `RELU` | 4x4 16-tile scatter, 8-pkt `OP_6c0`, RELU floor `19` | `e7f49f9f3a50` | `128/128` | **`0`** | `245` (`4.215`) | `420` (`1.283`) | `true` / `false` |
| `custom_8x64x64_nonpot003_relu6` | `(8, 64, 64)` | `s_out=0.03`, `zp=+13/-21`, `RELU6` | Dual-K 3D bcast, `k_passes=2`, `u_max[0]=1` FP32 ULP bit | `1603feb2d5c8` | `512/512` | **`0`** | `251` (`4.021`) | `447` (`1.372`) | `true` / `false` |
| `custom_16x64x64_nonpot007_none` | `(16, 64, 64)` | `s_out=0.07`, `zp=-17/+25`, `NONE` | `dive_scratch_196608`, `g=1` tile-0-before-3c0 scatter | `71a95aa9fe6b` | `1024/1024` | **`0`** | `427` (`4.844`) | `504` (`2.096`) | `true` / `false` |
| `custom_6x48x48_nonpot007_relu6` | `(6, 48, 48)` | `s_out=0.07`, `zp=+7/-15`, `RELU6` | `M%4==2` 6-pkt `OP_6c0`, `c3`+`c5` `VICA_CSR`, `CSR_040[0..1]`, active RELU6 `[-15, +71]` | `877d09171304` | `288/288` | **`0`** | `223` (`4.511`) | `443` (`1.310`) | `true` / `false` |
| `custom_3x32x24_nonpot005_relu` | `(3, 32, 24)` | `s_out=0.05`, `zp=-9/+11`, `RELU` | Odd `M=3`, non-PoT `N=24`, 15 tiles (`row_stride=5`) | `7e3daee6cece` | `72/72` | **`0`** | `238` (`4.288`) | `454` (`2.027`) | `true` / `false` |
| `custom_32x32x32_nonpot005_none` | `(32, 32, 32)` | `s_out=0.05`, `zp=+5/-7`, `NONE` | `M>16` 2D 64B scatter, `OP_3c0` cross-flit bit-16 overflow, `196608` scratch | `5a33cd5108a4` | `1024/1024` | **`0`** | `225` (`4.836`) | `487` (`2.465`) | `true` / `false` |
| `custom_10x32x32_unseen_relu6` | `(10, 32, 32)` | `s_out=0.07`, `zp=-13/-10`, `RELU6` | **Unseen `M=10` shape (not in 81-model corpus!)**, active RELU6 `[-10, +76]` | `8358cc36cb73` | `320/320` | **`0`** | `242` (`5.641`) | `430` (`1.522`) | `true` / `false` |

- Zero Google TPU libraries mapped in `tpu_baremetal_runner` across all 7 runs (`maps_has_google_tpu_libs == false`, only `libc.so`, `libdl.so`, `libm.so`, `libc++.so`, `libnetd_client.so`).
- Hardware telemetry confirmed active silicon execution (`thermal_zone24` `TPU`: `-274000 -> 30000 mC`).

---

## 4. Observed / Inferred / Unknowns Ledger

### Observed (Directly Proven)
- **81/81 bit-exact synthesis** of `chunk[0..6]`, `ext[0]` (`DIVE`), `ext[2]` (`params`), `ME.field_14`, and `dive_scratch_<size>` across all 81 compiled reference models spanning `M in 1..64`, `K in 4..512`, `N in 4..128`, non-multiple-of-4 edges (`1x5x5`, `1x17x16`, `1x16x17`), zero bias (`g21`), weight streaming (`K > 64`), and all activations (`NONE/RELU/RELU6`).
- **7/7 bit-exact live silicon execution** (`3,368/3,368` output bytes, `max_abs_err = 0`) on `rango` via both `tpu_direct_runner` and zero-Google-`.so` `tpu_baremetal_runner`, including unseen shape `(10, 32, 32)` and active RELU6 upper+lower saturation.
- Unified bit-77 shadow-CSR bank (`0x00..0x61`) and single shadow-state diffing across `HDR_DATA -> VICA_CSR -> CSR_040`.

### Inferred (Strong Consistency, Not Exhaustively Swept)
- Exact boundary where the compiler switches from `dive_scratch_192512` to `dive_scratch_196608` (`M * pad4(N) >= 1024`) matches 81/81 corpus models and executed cleanly on live `(16,64,64)` and `(32,32,32)`; larger outputs (`M*N >> 2048`) may require further 4 KiB scratchpad increments.
- `M >= 48` multi-tile paths with `N > 32` or `K > 32` beyond `(64, 32, 32)` and `N > 128` (`n_groups > 4`, where 4x4 grid has 16 tiles) are extrapolated from the closed-form equations and not yet silicon-tested.

### Unknowns / Out of Scope for Phase 4
- Per-channel weight quantization (`s_w[N]` vector) and INT4 (`W4A8`) / INT16x8 (`int16x8`) GEMM encodings (tracked in Phase 5).
- Multi-op fused graphs (e.g., GEMM + residual add or multi-layer dispatch inside a single `DGC0` `DIVE` function) and standalone RV64 DIVE interpreter regeneration (the static 638,977 B `ext[1]` ELF is preserved verbatim from the structural shell).
