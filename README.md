<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.jpg">
    <source media="(prefers-color-scheme: light)" srcset=".github/assets/banner-light.jpg">
    <img alt="FreeTPU — Bare-metal open-source runtime, VLIW ISA, and llama.cpp backend for Google Pixel Tensor TPU" src=".github/assets/banner-dark.jpg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-3ECF8E?style=flat-square&labelColor=11151B" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/target-Tensor%20G5%20(buenos__a0)-38BDF8?style=flat-square&labelColor=11151B" alt="Target: Tensor G5 buenos_a0">
  <img src="https://img.shields.io/badge/runtime-C%2B%2B17%20%2F%20Zero%20.so-F0B44C?style=flat-square&labelColor=11151B" alt="Runtime: C++17 Zero .so">
  <img src="https://img.shields.io/badge/backend-llama.cpp%20EDGETPU-38BDF8?style=flat-square&labelColor=11151B" alt="Backend: llama.cpp EDGETPU">
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-Build%20%26%20Clean--Room-3ECF8E?style=flat-square&labelColor=11151B" alt="CI: Build & Clean-Room Audit"></a>
</p>

<p align="center">
  <a href="#60-second-quickstart">Quickstart</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#verified-silicon--host-results">Benchmarks</a> &middot;
  <a href="#hardware-errata--w8a16-bf16-unlock">Hardware Errata</a> &middot;
  <a href="#documentation">ISA Docs</a> &middot;
  <a href="#clean-room-boundaries--firmware-extraction">Clean-Room Boundary</a>
</p>

<p align="center">
  <strong>Clean-room userspace runtime, closed-form VLIW ISA synthesizer, and <code>llama.cpp</code> GGML backend driving Google Pixel Tensor G5 TPU directly over <code>/dev/edgetpu-soc</code> with zero proprietary vendor <code>.so</code> dependencies.</strong>
</p>

<p align="center">
  <img alt="FreeTPU mascot breakdancing on its delidded IHS hood" src=".github/assets/freetpu-breakdance.gif" width="160">
</p>

---

## Architecture

<p align="center">
  <img src=".github/assets/architecture-diagram.svg" alt="FreeTPU execution pipeline from ggml-edgetpu through libopenbuenos and Litebuf VII to /dev/edgetpu-soc and the 4x4 serpentine PE mesh" width="100%">
</p>

| Layer | Component | Responsibility |
| --- | --- | --- |
| **Tile DSL** | [`freetpu_tile/`](docs/tile-dsl/README.md) | Clean-room `@freetpu.tile_kernel` frontend lowering one kernel (`matmul`, `GEMM+SiLU/GELU`, `SwiGLU`; `BF16`/`INT8`/`W8A16`) to `buenos_a0`, Cloud TPU Pallas (`tpu_custom_call`), and `cpu_ref` |
| **LLM Inference** | [`llama-cpp-edgetpu/`](llama-cpp-edgetpu/README.md) | `EDGETPU` GGML `ACCEL` backend (`MUL_MAT` & MoE `MUL_MAT_ID` `Q8_0`/`Q4_0`/`F32`, chained SwiGLU, `ROPE`/`GLU`, QKV `3->1` horizontal fusion, 1D bias `ADD`) |
| **Open Runtime** | [`libopenbuenos/`](libopenbuenos/include/openbuenos.h) | C11/C++17 static library: `1,286.4 MHz` DVFS + FW WakeLock, perf-core-pinned SPSC async dequant, clean-room 7-chunk PT `W8A16(FP16)` (`FP32` acc), adaptive spin, 2-tier DMA-BUF pool & LMKD single-copy weight cache, `M=1`/`M=64` GEMM + 11-chunk chained SwiGLU emitter |
| **Kernel Transport** | [`runtime/`](runtime/src/tpu_baremetal_runner.cpp) | Standalone bare-metal runner & UAPI/Litebuf VII headers (`0x0008 -> 0x00fe`, `/dev/edgetpu-soc` `0xED`) |
| **ISA Tooling** | [`tools/`](tools/emit_custom_gemm.py) | Pure-Python `DGC0` dissector/patcher, multi-tile `(M, K, N)` VLIW synthesizer, `0x640` minimax spline codec, `0xa40` chained fusion verifier, `INT4` packer |

---

## 60-Second Quickstart

Build `libopenbuenos`, verify strict C11/C++17 headers, and run the offline unit test suite (zero device or proprietary `.dgc0` files required):

```bash
git clone https://github.com/saariuslystoned/freetpu.git
cd freetpu

# 1. Build & test libopenbuenos (CMake + host mock mode)
cmake -S libopenbuenos -B libopenbuenos/build/cmake-host -DCMAKE_BUILD_TYPE=Release
cmake --build libopenbuenos/build/cmake-host -j
ctest --test-dir libopenbuenos/build/cmake-host --output-on-failure

# 2. Run clean-room Python VLIW ISA & 0x640 minimax spline self-tests
python3 tools/emit_custom_gemm.py --self-test
python3 tools/spline_0x640.py --self-test --no-emit
python3 tools/pack_int4.py

# 3. Run FreeTPU Tile DSL offline unit & parity tests (cpu_ref vs buenos_a0 mock vs Cloud TPU Pallas dry-run)
OPENBUENOS_MOCK=1 python3 -m freetpu_tile --self-test
OPENBUENOS_MOCK=1 python3 examples/tile_dsl/fused_gemm_silu.py --target all
```

To cross-build `libopenbuenos.a`, `tpu_baremetal_runner`, and `llama.cpp` (`llama-cli` / `llama-bench` / `test-backend-ops`) for Android ARM64 (`android-34`):

```bash
make -C libopenbuenos verify-header android verify-android
make -C runtime verify-baremetal
```

See [llama-cpp-edgetpu/README.md](llama-cpp-edgetpu/README.md) for patching upstream `ggml-org/llama.cpp` (`b12608`).

---

## Verified Silicon & Host Results

All hardware metrics below were measured on Google Pixel 10 Pro Fold (`rango`, Tensor G5 `buenos_a0`, `buenos.ko` ABI `1.4`) with `/proc/self/maps` confirming zero loaded Google/vendor `.so` libraries:

| Proof Suite | Scope | Verified Result |
| --- | --- | --- |
| **DVFS + FW WakeLock Silicon Unlock** | `1x2048x2048` & `64x1536x1536` (`PM=5` @ `1,286.4 MHz` + Litebuf `AcquireWakeLock`) | **`27.59 GB/s`** (`152 us` HW), **`983.6 INT8 GOPS` @ `177.2 mW` = `5.55 TOPS/W`** (`307 us` HW, `max_abs_err = 0`) |
| **Closed-Form Multi-Tile & Chained SwiGLU ISA** | `92` GEMM refs + `16` SwiGLU refs + `6` standalone non-PoT SwiGLU probes (`11` chunks, `sgs=1`) | **`92/92`** GEMM + **`16/16`** SwiGLU refs + **`6/6`** standalone SwiGLU probes bit-exact (`max_abs_err = 0`, up to **`952.6 GOPS`** / **`26.23 GB/s`**); **`20/20`** `libopenbuenos` host suites PASS; **`openbuenos_selftest --test-swiglu` `8/8` PASS live** (`nmse=2.14e-03` on M=64 Q4) |
| **`0x640` Minimax Splines & Chained Fusion** | `SiLU`/`GELU`/`exp` + `4` Mode-A (`+0 B`) + `3` multi-op `0xa40` probes (`MLP`, `SwiGLU`, `QKV`) | **`12,288/12,288`** spline bytes + **`3/3`** chained probes (`60.8–72.8 us`, `8/8` `ext[2]` slices bit-exact, `max_abs_err = 0`) |
| **Native PT `W8A16(FP16)` (`FP32` Acc) & Causal Huffman/`W4` Verdict** | `1x512..1x2048` (`--test-w8a16`, `openbuenos_selftest`) + `64x1536` 9-chunk PC compiler probe (FP16 intermediate) + `9` Huffman/`W4` silicon probes | **`8/8` single-row `--test-w8a16` PASS live** (`18.32 GB/s` at `1x2048` `229 us`, `relL2 = 2.13e-4`); separate `64x1536` 9-chunk PC compiler probe reached **`512.7 GOPS`**; causally proved hardware Huffman (`0x0a40 bit499` / `0x0cc0`) and 1:4 `W4` DMA unpack are **inactive on `buenos_a0`** |
| **`ggml-edgetpu` Backend Ops (Dense + MoE)** | Live `rango` `test-backend-ops -b EDGETPU` (`MUL_MAT` + `MUL_MAT_ID` MoE, `Q8_0` & `Q4_0`) | **`435/435` PASS** (`104/104` `MUL_MAT q8_0`, `104/104` `MUL_MAT q4_0`, **`227/227` `MUL_MAT_ID` MoE** = `115/115 q8_0` + `112/112 q4_0`), `cpu_fallback_slices = 0`, `DT_NEEDED = [libdl.so, libm.so, libc.so]` |
| **Live LLM Throughput, Residency & Energy (`llama-bench`)** | `SmolLM2-135M-Q8_0` & `Qwen2.5-1.5B-{Q8_0,Q4_0}` (`PM=5` @ `1,286.4 MHz`, `SPSC async dequant`, `MAPS_GOOGLE=0`) | **`SmolLM2 Q8 pp512 = 1,249.96 t/s`** (`-7.9%` SoC `mJ/tok` vs CPU, `tg64 = 34.23 t/s`); **`Qwen2.5 Q8 pp512 = 179.36 t/s`** (**`-62.7%` SS / `-55.3%` INT `mJ/tok` vs CPU** at `1.38 W`); **`Qwen2.5 Q4 pp512 = 180.37 t/s`** (**beats 4T CPU `160.2–176.3 t/s`**, `-39.8%` `mJ/tok`, warm `tg64 = 10.45 t/s`), `w_evict = 0` |

See [docs/PHASE8_PERF_REPORT.md](docs/PHASE8_PERF_REPORT.md) (and [Phase 7](docs/PHASE7_PERF_REPORT.md) / [Phase 6](docs/PHASE6_PERF_REPORT.md)) for SPSC overlapped NEON wins, causal Huffman/`W4`/`W8A16(FP16)` proofs, and full 18-case ODPM `mJ/tok` sweeps.

---

## Hardware Errata & `W8A16 (BF16)` Unlock

- **`DEQUANTIZE(INT4) -> FULLY_CONNECTED(FP32)` Compiler Deadlock ([LiteRT#12178](https://github.com/google-ai-edge/LiteRT/issues/12178))**: Live silicon ablation proves Tensor G5 (`buenos_a0`) has **no native 4-bit nibble-unpack DMA or systolic unit**. Descriptor `0x980` (`w8[21] = 1`, `w8[29:22] = zp`) is a **`1:2` hardware `UINT8 -> BFLOAT16` expander** (`BF16(float(u8 - zp))`, `16/16` exact at `398 us`), so emitting `1,024 B` of nibbles delivers `2,048 B` instead of `4,096 B` of `BF16` to the MAC and stalls execution indefinitely. **FreeTPU rule**: unpack `Q4_0` weights once at load time into cached `uint8` DMA-BUFs (same `1 B/weight` TPU DMA bandwidth as `Q8_0`), eliminating per-token CPU unpack overhead.
- **Native `W8A16 (BF16)` FP32-I/O GEMM Unlock**: Unpacking to `2,048 B` `uint8` (`w + 8`) + `128 B` per-channel `float32` scales at `+0x800` executes **`FP32 in -> BF16 MAC (1:2 u8->BF16 weight dequant) -> per-channel FP32 scale -> FP32 out`** natively on silicon in **`411–426 us`** (`32/32` bit-exact `FP32`, `max_abs_err = 0.0`). Local LLMs get full `BF16` activation precision (zero `INT8` clipping loss) at `Q8_0` DRAM bandwidth. See [docs/hardware-errata.md](docs/hardware-errata.md).

---

## Clean-Room Boundaries & Firmware Extraction

`freetpu` is strictly **clean-room source and specification only**:

- **Zero vendor binaries**: No proprietary `.so`, `.ko`, `.img`, `.apk`, `.gguf`, `.tflite`, `.dgc0`, or firmware blobs are included in this repository.
- **Zero vendor userspace dependencies**: `libopenbuenos`, `tpu_baremetal_runner`, and `ggml-edgetpu` talk directly to the Linux kernel (`/dev/edgetpu-soc` and `/dev/dma_heap/system`).
- **Host mock mode**: Setting `ob_config_t::mock_mode = 1` (or `OPENBUENOS_MOCK=1`) runs the full buffer pool, graph LRU cache, 16-credit async ring, VLIW emitter, and quantized math suites offline on macOS/Linux without TPU hardware.

### Extracting `dive_interp_rv64.bin` On-Device

Running synthesized `.dgc0` microkernels on physical Tensor G5 hardware requires the `638,977 B` RV64GCV DIVE VM interpreter image (`ext[1]`) from the user's own device. **Do not redistribute this binary.** Extract it locally on an unlocked Pixel using `libopenbuenos`:

```c
#include "openbuenos.h"
#include <stdlib.h>

size_t interp_size = 0;
ob_gemm_extract_dive_interp_from_dgc0("/path/to/local_compiled.dgc0", NULL, 0, &interp_size);
uint8_t *buf = (uint8_t *)malloc(interp_size);
ob_gemm_extract_dive_interp_from_dgc0("/path/to/local_compiled.dgc0", buf, interp_size, &interp_size);
```

Or point `libopenbuenos` / `ggml-edgetpu` directly at your local extracted image (or a local `.dgc0` file) at runtime:

```bash
export OPENBUENOS_DIVE_INTERP_PATH=/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin
```

---

## Documentation

- [FreeTPU Tile DSL (`@freetpu.tile_kernel`) Guide](docs/tile-dsl/README.md) — Single-source Python Tile IR lowering to `buenos_a0`, Cloud TPU Pallas (`tpu_custom_call`), and `cpu_ref`
- [Phase 8 Overlapped NEON, Causal Huffman/`W4`/`W8A16(FP16)` & ODPM Report](docs/PHASE8_PERF_REPORT.md) — SPSC perf-core async dequant (`SmolLM2 1,250 t/s`, `Qwen2.5 Q4` beats 4T CPU `pp512`, `Qwen2.5 Q8` `-62.7%` SoC `mJ/tok`), `8/8` `--test-swiglu` & `--test-w8a16`, causal Huffman/`W4` closure
- [Phase 8 Hardware Huffman, Sub-Byte Remapping & PT `W8A16(FP16)` Spec](docs/isa-gemm/phase8_huffman_w4a16_spec.md) — `buenos_a0_multimedia_huffman_config` diff, `0x0a40` 6-flit Huffman ISA, SMMU fault proof, native `FP16 + FP32` acc proof & C++17 recipe
- [Phase 7 Performance & Architecture Report](docs/PHASE7_PERF_REPORT.md) — 11-chunk chained SwiGLU, adaptive spin, `435/435` dense + MoE (`MUL_MAT_ID`), `SmolLM2 tg64 = 34.84 t/s` (`2.01x`), `Qwen2.5` prefill `-32.9%` to `-44.6%` SoC `mJ/tok`, `w_evict = 0` LMKD residency
- [Phase 7 In-SRAM Chained SwiGLU C++17 Spec](docs/isa-gemm/phase7_swiglu_cpp_spec.md) — `11`-chunk DIVE VM IR, `256 KiB` `0x940`/`0x980` parameter-ring formulas, `0x640` SiLU + `0x100` MUL CSR tables & `768 KiB` SRAM ceiling
- [Phase 6 Performance & Energy Report](docs/PHASE6_PERF_REPORT.md) — `1,286.4 MHz` DVFS + WakeLock unlock, `983.6 GOPS` (`5.55 TOPS/W`), batched `M=64` prefill, `SmolLM2`/`Qwen2.5` benchmarks & 18-case ODPM `mJ/tok` sweep
- [Phase 5B Live LLM Spec](docs/PHASE5_LLAMA_TPU_SPEC.md) — `Probe A` dynamic `param_slice`, `O(1)` weight cache, outlier peeling & baseline bring-up
- [DVFS, Power States & WakeLock Spec](docs/baremetal-vii/dvfs_power_states.md) — Corrected `+0x1c..+0x20` Litebuf wire fields, `TpuCorePowerState` (`0x03..0x19`), `MemoryPowerState` (`0x01..0x15`) & FW WakeLock (`union_type=4/5`)
- [Chained Multi-Op Fusion & W8A16(BF16) Spec](docs/isa-gemm/chained_fusion_and_w8a16_spec.md) — Zero-DRAM `0xa40` chaining (`MLP`, `SwiGLU` `0x100` MUL, `QKV` concat) & `W8A16(BF16)` scaling
- [Hardware & Driver Specs](docs/hardware-specs.md) — Tensor G5 (`buenos_a0`) geometry, `buenos.ko` ABI, rounding & SRAM layout
- [Hardware Errata & W8A16 (BF16)](docs/hardware-errata.md) — `DEQUANTIZE(INT4)` deadlock root cause, `Q4_0` caching rule & `W8A16 (BF16)` FP32 GEMM unlock
- [Litebuf VII Wire Protocol](docs/baremetal-vii/litebuf_vii_protocol.md) — Two-layer `0x0008 -> 0x00fe` command/response framing & 16-credit ring
- [Multi-Tile GEMM ISA](docs/isa-gemm/multitile_gemm_isa.md) — Shadow-CSR bank, `4x4` serpentine grid, 2D DMA rewind, `DIVE`/`DWS0`/`DGC0` layout
- [INT4 (`W4A8`) & Kernel Fusion](docs/isa-gemm/int4_and_fusion_spec.md) — `CSR 0x56`, 4-bit nibble packing (`50%` BW savings), Mode-A `GEMM + SiLU/GELU`
- [`0x640` Minimax Splines](docs/int4-splines/splines_silu_gelu_exp.md) — `208 B` `LoadCoefficientTables` codec & `SiLU`/`GELU`/`exp` Remez tables
- [DMA & TTU Bitfields](docs/isa-dma/dma_ttu_bitfields.md) — `0xa40` / `0x9c0` / `0x940` / `0x980` DMA descriptors & SRAM ring setup
- [Vector ALU ISA](docs/isa-vec/vector_alu_isa.md) — `stream125` FP32-RNE quantization & activation pipeline
- [Kernel UAPI Headers](docs/kernel-abi/README.md) — Clean-room `/dev/edgetpu-soc`, `/dev/gxp`, and GCIP IIF headers
- [`libopenbuenos` Guide](docs/libopenbuenos/README.md) & [`ggml-edgetpu` Guide](docs/llama-cpp-edgetpu/README.md)

---

## License

Licensed under the [Apache License 2.0](LICENSE).
