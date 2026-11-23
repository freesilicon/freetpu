# `ggml-cloudtpu` — Native C++17 `llama.cpp` Backend for Google Cloud TPUs

Clean-room Apache-2.0 GGML `GPU` backend (`CLOUDTPU`) targeting Google Cloud TPUs (`v5e`, `v6e`, `tpu7x`) directly over OpenXLA's C `PJRT_Api` (`dlopen(libtpu.so)` -> `GetPjrtApi()`).

**Zero Python, zero JAX, zero PyTorch, zero `vllm-tpu` at runtime (`MAPS_LIBPYTHON=0`).**

---

## Architecture (M1–M3)

| Component | File(s) | Responsibility |
| --- | --- | --- |
| **PJRT Runtime & Cache** | `pjrt_runtime.hpp`, `pjrt_runtime.cpp` | `dlopen(libtpu.so)` -> `GetPjrtApi()`, pure C++ `CompileOptionsProto` wire encoder, thread-safe LRU executable cache (`PJRT_LoadedExecutable*`), HBM buffer H2D/D2H with explicit untiled C-order `host_layout`, `/proc/self/maps` `libpython` audit, and built-in **Host Mock PJRT** (`CLOUDTPU_MOCK=1`) executing textual StableHLO IR on CPU for offline testing |
| **StableHLO Emitter** | `stablehlo_emitter.hpp`, `stablehlo_emitter.cpp` | Zero-dependency C++17 textual StableHLO MLIR emitter (`format="mlir"`) + deterministic structural cache key for `MUL_MAT` (`F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`), `FLASH_ATTN_EXT` (GQA + mask + ALiBi + softcap), `RMS_NORM`, `ROPE` (Normal & NeoX + YaRN + `rope_freqs`), `SILU`, `SWIGLU`, `SOFT_MAX`, `GET_ROWS`, `SET_ROWS` (`stablehlo.scatter`), `ADD`, `MUL`, `SCALE`, `CPY`, `RESHAPE`, `TRANSPOSE`, `SLICE` |
| **GGML Backend** | `ggml-cloudtpu.h`, `ggml-cloudtpu.cpp` | HBM-resident `ggml_backend_buffer` slot manager (lossless split HBM upload for `Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`, sub-slice view patching, `ggml_gallocr` liveness-aware subgraph fusion across `ggml_backend_sched` splits), and recursive `VIEW`/`PERMUTE`/`TRANSPOSE`/`RESHAPE` lowering |
| **PJRT Header** | `../libopentpu/third_party/openxla/pjrt_c_api.h` | Pinned Apache-2.0 OpenXLA `pjrt_c_api.h` (`0bfeeb476952fc0ae13dd38e27a9053cb1baa3ce`, ABI `0.116`, `struct_size=1144`) |

---

## Verified Live Silicon Results (`TPU v5e-1`, `MAPS_LIBPYTHON=0`)

All binaries (`test_cloudtpu_host`, `test-backend-ops`, `llama-cli`, `llama-bench`, `llama-server`) link with **`0` `libpython` entries in `ldd`** and execute with **`MAPS_LIBPYTHON=0` in `/proc/<pid>/maps`**.

### M2 Op Parity (`test-backend-ops -b CLOUDTPU`, `mock=0`)

| Suite | Passed | `MAPS_LIBPYTHON` | Notes |
| --- | --- | --- | --- |
| `SILU + SWIGLU` | `36/36` | `0` | `f32`/`f16`/`bf16`, packed & split GLU |
| `RMS_NORM` | `11/11` | `0` | `f32` |
| `ROPE` | `110/110` | `0` | Normal & NeoX, YaRN, `freq_factors`, non-multiple-of-8 (`M=52`) |
| `SOFT_MAX` | `61/61` | `0` | Scale, mask (`f16`/`f32`), ALiBi |
| `GET_ROWS` | `7/7` | `0` | `f32`/`f16`/`bf16`/`q8_0`/`q4_0`/`q4_K`/`q6_K` |
| `ADD + MUL` | `54/54` | `0` | `f32`/`f16`/`bf16`, periodic broadcast |
| `SCALE` | `12/12` | `0` | Scale + bias |
| `SET_ROWS` | `65/65` | `0` | `i32`/`i64` indices into `f32`/`f16`/`bf16` KV cache (`stablehlo.scatter`) |
| `CPY` | `9/9` | `0` | `f32`/`f16`/`bf16` conversions |
| `CONT` (permuted) | `20/20` | `0` | Arbitrary 4D `PERMUTE` lowering |
| `MUL_MAT` | `47/47` | `0` | `f32`/`f16`/`bf16`/`q8_0`/`q4_0`/`q4_K`/`q6_K`, GQA batch broadcast |
| `FLASH_ATTN_EXT` | `22/22` | `0` | GQA (`nr2=4`), `f16` KV cache views, causal mask, logit softcap |
| **Total** | **`454/454`** | **`0`** | **12/12 suites passed on live `TPU v5 lite` (`libtpu.so`)** |

### M3 End-to-End GGUF Inference (`llama-bench -ngl 99 -fa 1`, `llama-cli`, `llama-server`)

| Model (GGUF) | Quant | Size | `pp128` (tok/s) | `pp512` (tok/s) | `tg64` (tok/s) | `llama-cli` / `llama-server` |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `4,944.70` | `16,113.17` | `39.17` | Coherent English; `llama-server` `/v1/chat/completions` (`17*23=391`), `/proc/<pid>/maps` `libpython=0` |
| `Qwen2.5-1.5B-Instruct` | `Q8_0` | `1.76 GiB` | `5,445.35` | `19,820.42` | `42.24` | Coherent English |
| `Meta-Llama-3.1-8B-Instruct` | `Q4_K_M` | `4.58 GiB` | `1,957.64` | `4,684.60` | `33.25` | Coherent English; `llama-server` `/v1/chat/completions` (`17*23=391`), `/proc/<pid>/maps` `libpython=0` |
| `Meta-Llama-3.1-8B-Instruct` | `Q8_0` | `7.95 GiB` | `1,910.83` | `4,207.50` | `35.20` | Coherent English |

---

## Quick Start

### 1. Run Standalone Host Mock Tests (No TPU or `llama.cpp` Checkout Required)
```bash
./llama-cpp-cloudtpu/build.sh standalone
```

### 2. Build & Test Overlaid on `ggml-org/llama.cpp` (`b12608`)
```bash
git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp llama-cpp-cloudtpu/upstream/llama.cpp-b12608
./llama-cpp-cloudtpu/build.sh host

# Host Mock verification against ggml-cpu:
CLOUDTPU_MOCK=1 ./llama-cpp-cloudtpu/build-host/bin/test-backend-ops -b CLOUDTPU \
  -o "MUL_MAT,RMS_NORM,ROPE,SOFT_MAX,GET_ROWS,ADD,MUL,CPY,SILU,SWIGLU"
```

---

## Runtime Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLOUDTPU_MOCK` | `0` on Linux, `1` on macOS/Windows | `1` runs in-process Host Mock PJRT executing emitted StableHLO IR |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | Auto-resolved `libtpu.so` | Explicit path to `libtpu.so` (or future clean-room `libopentpu.so`) |
| `CLOUDTPU_FUSE_GRAPHS` | `1` | `1` fuses contiguous supported GGML nodes into one StableHLO executable |
| `CLOUDTPU_CACHE_CAP` | `4096` | LRU capacity of compiled `PJRT_LoadedExecutable*` cache |
| `CLOUDTPU_MIN_ELEMENTS` | `0` | Minimum `MUL_MAT` element work threshold before offloading to TPU |
| `CLOUDTPU_VERBOSE` | `0` | `1` logs backend init, device HBM stats, and executable cache hit rate |
| `GGML_DISABLE_CLOUDTPU` | unset | Set `1` to disable `CLOUDTPU` device registration |
