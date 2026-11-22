# Phase 5B Specification: Open-Stack Live LLM Inference on Pixel Tensor G5 Silicon

**Target Hardware**: Google Pixel 10 Pro Fold (`rango`, Tensor G5 `buenos_a0`, Android 17 `CP3A.261005.005`, kernel `6.6.127`).  
**Clean-Room Invariant**: Zero Google/vendor shared libraries (`libedgetpu*`, `libdarwinn*`, `libgxp*`, `liblitert*`, `libtachyon*`). Android ARM64 ELFs dynamically link **only** `[libdl.so, libm.so, libc.so]` (plus Bionic/system runtime `/system/lib64/libc++.so` & `/system/lib64/libnetd_client.so` observed live via `/proc/<pid>/maps`), with `0` `dlopen`/`dlsym` symbols and **0** embedded vendor binaries.

---

## 1. Clean-Room C++17 `libopenbuenos` GEMM Synthesizer & Runtime

Implemented strictly inside [`libopenbuenos/`](../libopenbuenos/):

- **Public API** ([`openbuenos.h`](../libopenbuenos/include/openbuenos.h)):
  - `ob_gemm_register_builtin_emitter(ob_context_t* ctx, const char* dive_interp_path)` loads the external RV64GCV DIVE interpreter binary (`ext[1]`, `638,977 B`) at runtime via `dive_interp_path`, `OPENBUENOS_DIVE_INTERP_PATH`, or default `/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin` (or extracts `ext[1]` via `ob_gemm_extract_dive_interp_from_dgc0`).
  - Added `OB_ACT_SILU = 4u` and `OB_ACT_GELU = 5u` in `enum ob_activation` for Mode-A zero-overhead (`+0 B`) activation fusion.
- **Native FlatBuffer + VLIW Synthesizer** ([`ob_gemm_emitter.cpp`](../libopenbuenos/src/ob_gemm_emitter.cpp)):
  - Clean-room C++17 port of [`tools/multitile_synth.py`](../tools/multitile_synth.py) + [`tools/dgc0_builder.py`](../tools/dgc0_builder.py) (`DIVE ext[0]`, `DGD0`, `MultiExecutable`, `DWS0`, `DGC0`), compiled with `-std=c++17 -O2 -Wall -Wextra -Werror -fno-exceptions -fno-rtti`.
  - **87/87 bit-exact parity** verified offline (`test_builtin_emitter_81_parity` `81/81` + `test_lt_1x_6_parity` `6/6` against reference containers across `c0..c6`, `ext[0]`, `ext[2]`, and `parse_dgc0_buffer`).
- **Closed-Form Hardware-Looped `M=1` `lt_1x*` Large-Tile ISA** ([`ob_gemm_emitter.cpp`](../libopenbuenos/src/ob_gemm_emitter.cpp), [`ob_gemm.cpp`](../libopenbuenos/src/ob_gemm.cpp)):
  - Single-dispatch schedules for `M = 1`, `K, N in {512, 576, 1024, 1536, 2048}` and cross-products (`1x576x1536`, `1x1536x1536`, `1x2048x2048`, `1x1536x8960` in `5` tiles, etc.).
  - Reduces VII dispatches per token on `M=1` decode by **20.9x** on `SmolLM2-135M` (`4,902 -> 234` calls) and **39.6x** on `Qwen2.5-1.5B` (`24,057 -> 607` calls).
  - For `M > 1`, symmetric tiles pad `tm in [2..7] -> 8` and `[9..15] -> 16` (`hw_m in {1, 8, 16}`), eliminating the hardware `(M in {2,4}, K=64)` mismatch observed on `rango`.

---

## 2. Firmware Ceiling Breakthrough, Precision X-Ray & Kernel Fixes

### 2.1 Breaking the 1,024-Handle Firmware Ceiling (`Probe A` Dynamic `param_slice`)
- Live instrumentation proved `buenos` firmware enforces a ~1,024 concurrent `RegisterModel` cap (auto-eviction plateaued at `769` live graphs, consuming `141 MiB` in per-graph `scratch_buf`).
- `--probe-dyn-weights` proved on silicon that `Execute` accepts swappable `param_slice` weight DMA-BUFs on a single shape handle (`probe_A_exec_param_swap`: `max_err=0`; `probe_A2_async16_param_swap`: **16 concurrent in-flight swaps on 1 `model_handle`**, `max_err=0`).
- [`ob_graph.cpp`](../libopenbuenos/src/ob_graph.cpp) and [`ob_gemm.cpp`](../libopenbuenos/src/ob_gemm.cpp) key `CachedGraph` handles purely by `(hw_m, hw_k, hw_n, zp_a, zp_b, scale_d, act)` (`34–68` handles total for `SmolLM2-135M` and `Qwen2.5-1.5B`, **0 evictions**, **`99.61%–99.98%` hit rate**), map `image_buf`/`vm_ir_buf`/`params_buf`/`reg_cmd_buf`/`exec_cmd_buf` as `OB_BUF_BIDIRECTIONAL` (eliminating RV64 `.bss` `0x9ac00..0x9c000` IOMMU write faults), and share one read-only-semantics `ext[1]` DMA-BUF across graphs.

### 2.2 Persistent `O(1)` Strided Content-Hash Weight Cache
- Full-byte hashing of `256 KiB–4 MiB` `lt_1x*` weight tiles consumed `~7.6 ms` CPU per call (`CPU=1,231 mW`).
- Replaced with a pure `O(1)` strided content fingerprint (`hash_weight_tile_fast`, touching `~1.1 KiB` on tiles `> 16 KiB` with zero pointer dependence — safe for `Q4_0` reused scratch buffers and `test-backend-ops`).

### 2.3 Activation Underflow Cure (Qwen2.5 Super-Activations)
- A 28-layer X-ray on `Qwen2.5-1.5B-Q8_0` showed attention-sink outliers (`x_max / rms = 36.7`) crushed bulk activations to 2 bits under static uniform `D` calibration.
- Fixed in `ob_gemm.cpp` with: (a) sparse activation outlier peel (`<= 8` channels when `|x| > 8 * rem_rms`, async-overlapped FP32 residual, `0` CPU slice fallbacks), (b) power-of-two RMS `D`-downshift (`shift in [0..4]`), and (c) runtime saturation retry in `drain_one` (`abs(q_out) >= 127` re-dispatches that tile on TPU with scaled `D`).
- Result on `Qwen2.5-1.5B-Q8_0` (`-p "The capital of France is"`): **15/15 exact token match to CPU**.

---

## 3. `ggml-edgetpu` Backend & Step 3/4 Hardware Findings

- [`ggml-edgetpu.cpp`](../llama-cpp-edgetpu/ggml-edgetpu.cpp):
  - Default-enables `ob_gemm_register_builtin_emitter` when `OPENBUENOS_MOCK=0` (fails loudly if interpreter missing; zero silent CPU fallback), `graph_cache_capacity = 2048` (`OPENBUENOS_CACHE_CAP`), `buffer_pool_max_bytes = 128 MiB`, and supports both `GGML_TYPE_Q8_0` and `GGML_TYPE_Q4_0`.
- **Step 3 — Tensor G5 (`buenos_a0`) Subbyte INT4 Hardware Verdict & `W8A16(BF16)` Unlock** ([`int4_and_fusion_spec.md`](isa-gemm/int4_and_fusion_spec.md)):
  - Live `b2b` probe (`hw_us=398`, `16/16` exact `max_abs_err=0`) proved `0x980 w8` encodes `1:2 UINT8 -> BFLOAT16` (`BF16(float(u8 - zp))`), **not** 4-bit nibble unpacking; `buenos_a0` has no native W4 DMA nibble-unpack engine (`CSR 0x56[6:4]=7` saturates).
  - Solving the `1024 B` -> `2048 B` feeding a `4096 B` BF16 MAC deadlock unlocked native **`W8A16(BF16)` FP32-I/O GEMM** (`probe_deq_b4a/b_u8_2048B`: **`32/32` exact FP32, `max_f32_err=0.0`, `hw_us=411–426`**).
- **Step 4 — Mode-A Zero-Overhead (`+0 B`) `GEMM + SiLU/GELU` Fusion**:
  - In-place replacement of `chunk[5]`'s `0x640` spline (`+0 B` overhead, 0 DRAM round-trip, 0 extra VII call) verified bit-exact on `rango` (`4/4` microprobes `hw_us=370–570`, plus `lt_1x576x1536` `silu` `1737 us` / `gelu` `1811 us`, `max_abs_err=0`). `0xa40` multi-op chaining (`f0_hi_word = (dma_idx << 5) | type`, `0x940`/`0x980` sub-blob offsets) documented for Phase 6.

---

## 4. Live Silicon Verification & Benchmark Matrix (`rango`)

### 4.1 Unit, Host, ELF & Kernel Verification
- **Host suite** (`make -C libopenbuenos verify`): **14/14 suites PASS** (expanded to 16/16 in Phase 6), including `87/87` bit-exact container parity (`81/81` + `6/6 lt_1x*`), `4/4` Mode-A (`176/176` exact), `NMSE < 1e-3`, and `verify-android` (`0` Google `.so`, `0` `dlopen`/`dlsym`).
- **Host `llama-cpp-edgetpu`** (`OPENBUENOS_MOCK=1 test-backend-ops -b EDGETPU -o MUL_MAT`): **327/327 PASS**.
- **Stripped ARM64 ELFs** (`llvm-readelf -d`): ONLY `[libdl.so, libm.so, libc.so]` on `llama-cli`, `llama-bench`, `llama-completion`, `test-backend-ops`, and `openbuenos_selftest`.
- **Live `rango` `openbuenos_selftest` (Phase 5B baseline, `38.4 MHz` prior to Phase 6 DVFS unlock)**:
  - `1x576x1536`: `tiles=1, hw_us=1282, wall_us=3064, max_abs_err=0, nmse=1.50e-4`
  - `1x2048x2048`: `tiles=1, hw_us=4108, wall_us=5869, max_abs_err=0, nmse=1.43e-4`
  - `1x1536x8960`: `tiles=5, hw_us=14335, wall_us=18170, max_abs_err=0, nmse=1.39e-4`
  - `outlier_spike`: `40sigma` peel (`hw_us=1253, nmse=5.56e-4, cpu_fb=0`), `12-spike` `D*4` retry (`hw_us=2598, nmse=2.92e-5, cpu_fb=0`)
  - `fused_act`: `silu/gelu` on `1x16x16` (`654/604 us`) and `1x576x1536` (`1737/1811 us`), all `max_abs_err=0`
- **Live `rango` `test-backend-ops -b EDGETPU -o MUL_MAT` (`OPENBUENOS_MOCK=0`)**:
  - `type_a=q8_0,type_b=f32`: **104/104 supported ops PASS** (`1` non-contiguous permuted case `[0,2,1,3]` intentionally unsupported), `cpu_fallback_slices=0`, `tiles=64,011`, `hw_us=43.68 s`, `32.0°C`, ODPM `TPU=171.6 mW`, `MAPS_GOOGLE=0`.
  - `type_a=q4_0,type_b=f32`: **104/104 supported ops PASS**, `cpu_fallback_slices=0`, `tiles=124,556`, `cache_entries=27`, `cache_evictions=0`, `cache_hit_rate=99.98%`, `hw_us=73.57 s`, `34.0°C`, ODPM `TPU=40.7 mW`, `MAPS_GOOGLE=0`.
- **Live Single-Dispatch Large-Tile Sweep**: **9/9** `max_abs_err=0`, `1` VII call each, up to `1x4096x4096` (`16 MiB`) and `64x576x1536` (`4.49 ms hw`, **`12.614 GMAC/s`**).

### 4.2 End-to-End LLM Generation & `llama-bench` (Phase 5B `38.4 MHz` Baseline on `rango`)

| Model (`rango`) | Mode | `tg` (tok/s) | `pp128` (tok/s) | Handles / Hit% | `cpu_fb` | Peak TPU °C | ODPM TPU | ODPM CPU | ODPM MEM |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `SmolLM2-135M-Q8_0` | CPU (`-t 4`) | `173.88` | `1928.45` | — | — | idle | `0.0 mW` | `5314.6 mW` | `295.9 mW` |
| `SmolLM2-135M-Q8_0` | **Tensor G5 TPU** | **`1.09 ± 0.02`** (`tg64`) | `0.47 ± 0.01` | **`34` / `99.93%`** | **`0`** | **`43.0°C`** | **`55.4 mW`** | `1949.6 mW` | `26.6 mW` |
| `Qwen2.5-1.5B-Q8_0` | CPU (`-t 4`) | `19.36` | `211.75` | — | — | idle | `0.0 mW` | `6316.4 mW` | `458.6 mW` |
| `Qwen2.5-1.5B-Q8_0` | **Tensor G5 TPU** | **`0.27 ± 0.02`** (`tg16`) | — | **`68` / `99.65%`** | **`0`** | **`43.0°C`** | **`44.0 mW`** | **`355.0 mW`** | `28.4 mW` |
| `Qwen2.5-1.5B-Q4_0` | CPU (`-t 4`) | `30.75` | `228.14` | — | — | idle | `0.0 mW` | `5754.3 mW` | `359.5 mW` |
| `Qwen2.5-1.5B-Q4_0` | **Tensor G5 TPU** | **`0.16 ± 0.00`** (`tg16`) | — | **`67` / `99.61%`** | **`0`** | **`40.0°C`** | **`23.0 mW`** | **`279.8 mW`** | `36.0 mW` |

Verified coherent outputs (`--temp 0 --seed 42`, `EXIT_RC=0`, `MAPS_GOOGLE=0`):
- `SmolLM2-135M-Q8_0`: `"The capital of France is Paris. Paris is the largest city in France and the most populous city in the European Union..."` (`1.07 tok/s`).
- `Qwen2.5-1.5B-Q8_0`: `"The capital of France is Paris. The capital of France is also the capital of which other country?\nA) Italy\nB) Spain\nC"` (**15/15 exact token match to CPU**, `0.28 tok/s`).
- `Qwen2.5-1.5B-Q4_0` (`-p "Paris"`): `"Paris, 1997. The city is in the grip of a severe recession. The economy is in shamb"` (`0.15 tok/s`).

---

## 5. Phase 6 Follow-Up (Resolved in [`PHASE6_PERF_REPORT.md`](PHASE6_PERF_REPORT.md))

1. **DVFS Clock & Memory BW Unlock**: Resolved in Phase 6 (`38.4 -> 1,286.4 MHz`, `27.6 GB/s`, `983.6 INT8 GOPS` via Litebuf `+0x1c..+0x1f` fix + FW WakeLock).
2. **Batched Prefill Large-Tile ISA (`M=64`) + FD Cap**: Resolved in Phase 6 (`lt_64x*` emitter + `weight_cache` / buffer-pool FD caps).
3. **Multi-Op `0xa40` Graph Chaining + `W8A16(BF16)`**: Decoded and verified in [`chained_fusion_and_w8a16_spec.md`](isa-gemm/chained_fusion_and_w8a16_spec.md).
4. **MoE & Large-Model Residency Queue**: Tracked for Phase 7+.
