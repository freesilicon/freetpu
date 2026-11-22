# FreeTPU Silicon Cheatsheet (`buenos_a0` vs `v5e-1` / `v6e-1`)

Compact hardware, memory-hierarchy, and errata reference for kernel authoring and debugging.

---

## 1. Architecture & Memory Hierarchy Comparison

| Dimension | Pixel Tensor G5 (`buenos_a0` / `laguna`) | Cloud TPU `v5e-1` (`TPU v5 lite`) | Cloud TPU `v6e-1` (`Trillium`) |
| --- | --- | --- | --- |
| **Kernel Transport** | `/dev/edgetpu-soc` (`buenos.ko` ABI `1.4`, Litebuf VII, `16`-credit `eventfd` ring) | `/dev/vfio/0` (`vfio-pci`, PCI `0x1ae0:0x0063`, `128` MSI-X) + `libtpu.so` PJRT | `/dev/vfio/*` + `libtpu.so` PJRT (`colab --tpu V6E1`) |
| **Compute Topology** | `4x4` (`16` tiles) serpentine PE mesh (`SNAKE_16 = [0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]`), `2` RV64GCV DIVE control cores | `4` MXUs (`128x128` systolic), `1` core per chip | Single-chip Trillium (`colab run --tpu V6E1`) |
| **Clock / OPP** | `1.12 GHz` nominal; **`1,286.4 MHz`** peak (`PM=5` `0x19` + FW WakeLock); `38.4 MHz` sleep (`0x03`) | Datacenter fixed OPP | Datacenter fixed OPP |
| **On-Chip SRAM** | **`768 KiB`** Narrow Memory (NM) SRAM / core (`1 MiB` unit memory) | **`128 MiB`** VMEM total (**`16 MiB`** default per-kernel Pallas `vmem_limit_bytes`), `1 MiB` SMEM | High-capacity VMEM (keep Pallas tiles `<= 16 MiB` for v5e/v6e portability) |
| **Off-Chip Memory** | Unified phone LPDDR (`27.59 GB/s` measured @ `1x2048x2048`), DMA-BUF 36-bit IOVA | **`15.748 GiB`** usable HBM2e (`820 GB/s` peak; **`655.55 GB/s`** triad / **`684.27 GB/s`** 32L `W8A16` measured) | **`32 GiB`** HBM |
| **Peak Matrix Compute** | `983.6 INT8 GOPS` @ `177.2 mW` (`5.55 TOPS/W`, `64x1536x1536`) | `197 TF/s` BF16 (**`180.36 TF/s`** measured C++ PJRT), `394 TOPS` INT8 (`350.69` measured) | Next-gen Trillium BF16/INT8 |
| **Small-Op Floor** | **`60.8–72.8 us`** chained `0xa40` fusion (`152 us` `1x2048x2048` `PM=5` + FW WL) | **`115–135 us`** sync PJRT floor (`126.94 us` median); **`32.00 us`** amortized (`100x` async pipeline) | Similar PJRT dispatch regime |
| **Rounding & Activations** | IEEE-754 **FP32 RNE** (`np.rint` / `std::nearbyint`) + `zp_out`; `0x640` 8-seg degree-4 Horner splines | `BF16` MXU inputs with **`FP32` accumulation** (`preferred_element_type=jnp.float32`) | `BF16` MXU + `FP32` accumulation |

---

## 2. Pixel Tensor G5 (`buenos_a0`) Tiling & ISA Rules

1. **Tile Limits & Alignment**:
   - `libopenbuenos` default tile ceilings: `M <= 64`, `K <= 512`, `N <= 256` (fits `768 KiB` NM SRAM; `M=64` batched prefill supports `K + N <= 3072`).
   - `ext[2]` parameter packing aligns `K` to 4 (`pad4(K)`) and `N` to `n_blk = 32 if N > 32 else max(8, ceil8(N))` in `[K/4, n_blk, 4]` order with `(w + 128) & 0xFF`.
2. **Exact Arithmetic (`FP32-RNE`, not fixed-point RHU)**:
   - Tensor G5 `stream125` computes `val = round_half_to_even(f32(acc) * f32(f32(s_in * s_w) / s_out)) + zp_out` clamped to `[qmin, qmax]`.
   - Always verify CPU references with `np.rint` / `round()` in `float32`, not TFLite's `(q_mul, q_shift)` fixed-point multiplier.
3. **`0x640` Minimax Splines (`SiLU`, `GELU`, `exp`)**:
   - `208 B` (`1664-bit`) `LoadCoefficientTables` holds 8 segments x 5 `float32` Horner coefficients `[c0..c4]` + 7 ascending `float32` breakpoints (`tail = 0x7804` for deg-4).
   - Output quantization `u(x) = f(x) / s_out + (zp_out + 128)` folds directly into `c0..c4` with `+0 B` overhead.

---

## 3. Tensor G5 Hardware Errata & Unlocks

- **Erratum 1 — `DEQUANTIZE(INT4) -> FULLY_CONNECTED(FP32)` Deadlock ([LiteRT#12178](https://github.com/google-ai-edge/LiteRT/issues/12178))**:
  - `buenos_a0` has **no native 4-bit nibble-unpack DMA or systolic unit**.
  - `CSR 0x56[6:4] = 7` saturates nibble-packed payloads; DMA descriptor `0x980` (`w8[21] = 1`, `w8[29:22] = zp`) is a **`1:2` `UINT8 -> BFLOAT16` expander** (`BF16(float(u8 - zp))`). Feeding `1,024 B` of nibbles delivers `2,048 B` instead of `4,096 B` to the `BF16` MAC and stalls execution until fence timeout.
  - **Fix**: Store `Q4_0` compactly on disk/host RAM and unpack once at model load into cached `uint8` DMA-BUFs (`1 B/weight`).
- **Unlock — Native `W8A16 (BF16)` FP32-I/O GEMM**:
  - Unpacking to `uint8` (`w + 8` in `[K/2, N, 2]`, `0x980 w8[21]=1, zp=8`) + contiguous per-channel `float32` scales at `+0x800` executes **`FP32 in -> BF16 MAC -> per-channel FP32 scale -> FP32 out`** in **`411–426 us`** (`1x64x32`, `32/32` bit-exact `FP32`) at `1 B/weight` DMA traffic.
- **Erratum 2 — Litebuf VII DVFS Wire Shift (`38.4 MHz` Trap)**:
  - `VII_CMD #2` (`struct litebuf_vii_execute`) carries `fw_priority` (`+0x1c`), `tpu_core_power_state` (`+0x1d`), `memory_power_state` (`+0x1e`), `qos_class` (`+0x1f`).
  - Default sleep state `0x03` runs at `38.4 MHz` (`~5.9 ms` for `1x2048x2048`); setting `PM=5` (`core=0x19` `1,286.4 MHz`, `mem=0x15`) + Firmware `AcquireWakeLock` (`union_type=4`, `bitset=0x04`) drops `1x2048x2048` to **`152 us` HW** (`38.6x`).

---

## 4. Cloud TPU (`v5e-1` / `v6e-1`) Pallas/Mosaic Pitfalls

1. **VMEM OOM (`> 16 MiB`)**:
   - Double-buffered `(2, bm, bk)` + `(2, bk, bn)` `bf16` input tiles, `(bm, bn)` `f32` accumulators, and double-buffered `(2, bm, bn)` `bf16` output windows must fit inside `16 MiB` (`1024x2048x256` needs `19 MiB` and overflows; `1024x1024x256` needs `10 MiB` and `1024x512x512` needs `7 MiB`, fitting cleanly and hitting `135.5 TFLOP/s` on `v5e-1`).
2. **PJRT Dispatch Floor (`115–135 us`)**:
   - Both Python (`jax.jit`) and Zero-Python C++17 PJRT hit `~125 us` `sync_ready_event` per tiny op (`1x64x32`). Fuse activations (`GEMM+SiLU`, `SwiGLU`, `RMSNorm`) or pipeline async launches (`32 us` amortized).
3. **Colab Allocation Errors (`HTTP 412 / 400`)**:
   - `REASON_NO_BACKENDS` or `REASON_TOO_MANY_ACTIVE_ASSIGNMENTS` on `V6E1` during pre-allocation should fall back to `V5E1` (handled automatically by `tools/colab_tpu_bench.py`). Never fall back after `[colab] Session READY`.
4. **CompilerParams & MemorySpace Portability**:
   - Recent JAX exposes `pltpu.CompilerParams`; older builds (e.g., JAX `0.5.3`) expose `pltpu.TPUCompilerParams` without optional newer kwargs such as `disable_bounds_checks`. Resolve `getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams")` and pass optional kwargs only when present in `inspect.signature` / `dataclasses.fields`.
   - Use `pl.BlockSpec(memory_space=pl.ANY)` (supported on both JAX `0.5.3` and newer JAX) rather than `pltpu.HBM` for explicit HBM DMA inputs, and express sigmoid as `1.0 / (1.0 + jnp.exp(-x))` inside Pallas kernels instead of `jax.nn.sigmoid` (`logistic_p`, unsupported on JAX `0.5.3` Mosaic).
