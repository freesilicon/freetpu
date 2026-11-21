# Tensor G5 (`buenos_a0`) Hardware Errata & `W8A16 (BF16)` Unlock

Clean-room silicon characterization on Google Pixel Tensor G5 (`buenos_a0`), executed directly against `/dev/edgetpu-soc` with zero proprietary userspace `.so` dependencies.

---

## 1. `DEQUANTIZE(INT4) -> FULLY_CONNECTED(FP32)` Compiler Deadlock Errata

- **Upstream issue**: [`google-ai-edge/LiteRT#12178`](https://github.com/google-ai-edge/LiteRT/issues/12178)

### Root Cause (Silicon-Proven)

Tensor G5 (`buenos_a0`) has **no native 4-bit nibble-unpack DMA or systolic unit**:

1. **`CSR 0x56[6:4] = 7` is non-functional for nibble-packed weights on `buenos_a0`**:
   - Driving nibble-packed payloads with `CSR 0x56 = 0x0010070` saturates every output channel (`[127] * 16`).
   - Clearing `CSR 0x56[6:4]` (`7 -> 0`, `CSR 0x56 = 0x0010000`) with unpacked `uint8` weights `w + 8` executes accurately (`15/16` exact, `max_abs_err = 1`).
2. **DMA descriptor `0x980` word `w8` implements `1:2` `UINT8 -> BFLOAT16` expansion (`1 B -> 2 B`), NOT 4-bit nibble unpacking**:
   - Causal live-silicon ablation (`1x16x16`, `P = 128 B` payload, `hw_us = 398`, `16/16` bit-exact `max_abs_err = 0`) proves `0x980 w8` encodes only two fields on `buenos_a0`:
     - **`w8[21] = 1`**: enables `1:2` `UINT8 -> BFLOAT16` expansion on DMA fetch (`1 B u8 -> 2 B BF16`).
     - **`w8[29:22]`**: 8-bit unsigned `zero_point` (`0..255`), evaluated prior to widening:
       $$\mathrm{BF16}(\mathrm{float}(u_8 - zp))$$
3. **Why stock `DEQUANTIZE(INT4) -> FULLY_CONNECTED(FP32)` stalls forever (`Wait rc=-1` / `poll(resp_efd)` timeout)**:
   - On a `1x64x32` `DEQUANTIZE(INT4) -> FULLY_CONNECTED(FP32)` graph, the stock compiler emitted **`1,024 B`** of packed 4-bit nibbles (`64 * 32 / 2` weights) while programming `0x980 w8 = 0x02200000` (`w8[21] = 1`, `zp = 8`).
   - Because `buenos_a0`'s `0x980` expander widens `1 B -> 2 B` (`1:2`), `1,024 B` delivered only **`2,048 B` (`1,024` `BF16` elements)** instead of the **`4,096 B` (`2,048` `BF16` elements)** consumed by the `64x32` `BF16` systolic MAC (`CSR 0x56 = 0x0000026`), starving the systolic array indefinitely until fence timeout.

### FreeTPU Design Rule for `Q4_0` & Sub-Byte Models

- Store and load `Q4_0` (or custom 4-bit) weights compactly on disk and in host RAM.
- Unpack 4-bit weights **once at model load time** into cached `uint8` DMA-BUFs (the same `1 B/weight` TPU DMA footprint as `Q8_0`), completely eliminating per-token CPU unpack overhead during autoregressive decode.

---

## 2. Native `W8A16 (BF16)` FP32-I/O GEMM Unlock

Expanding the `1x64x32` `DEQUANTIZE -> FULLY_CONNECTED` weight payload from `1,024 B` nibbles to `2,048 B` unpacked `uint8` (`w + 8` in `[K/2, N, 2]` order, `0x980 w8 = 0x02200000`) plus `128 B` contiguous per-channel `float32` scales at `+0x800` satisfies the `4,096 B` `BF16` MAC contract on real silicon:

```text
FP32 activations ─▶ TPU VICA FP32→BF16 ─▶ BF16 Systolic MAC ◀─ 0x980 1:2 u8→BF16 DMA (w + 8, zp=8)
                                                 │
                                                 ▼
FP32 outputs     ◀─ Per-channel FP32 rescale (+0x800, 128 B)
```

### Verified Silicon Measurements (`1x64x32` `W8A16 (BF16)` Probes)

| Configuration | Payload | Hardware Latency | Output Parity (`32 x FP32`) |
|---|---|---:|---|
| Baseline stock packing | `1,024 B` nibbles | Deadlock (`timeout`) | N/A |
| Unpacked `uint8` (`w + 8`), default schedule | `2,048 B u8` + `128 B fp32` | **`426 us`** | **`32/32` bit-exact (`max_abs_err = 0.0`)** |
| Unpacked `uint8` (`w + 8`), `c[5]->c[7]` scale ordering | `2,048 B u8` + `128 B fp32` | **`411 us`** | **`32/32` bit-exact (`max_abs_err = 0.0`)** |

### Why This Matters for Local LLM Inference

- **Full `BF16` activation precision**: activations stay `FP32` at the TPU boundary and convert to `BF16` on-chip — avoiding `INT8` activation dynamic-range clipping on outlier transformer channels.
- **Native per-channel `FP32` weight rescaling**: hardware multiplies accumulator results directly by per-channel `float32` scales without CPU quantize/dequantize passes around every projection.
- **Same `1 B/weight` DRAM bandwidth as `Q8_0`**: uses the exact same `1 B/weight` TPU DMA traffic while upgrading from `W8A8` (`INT8`) to `W8A16 (BF16)` precision.
