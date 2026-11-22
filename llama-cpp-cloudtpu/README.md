# `ggml-cloudtpu` — Native C++17 `llama.cpp` Backend for Google Cloud TPUs

Clean-room Apache-2.0 GGML `GPU` backend (`CLOUDTPU`) targeting Google Cloud TPUs (`v5e`, `v6e`, `tpu7x`) directly over OpenXLA's C `PJRT_Api` (`dlopen(libtpu.so)` -> `GetPjrtApi()`).

**Zero Python, zero JAX, zero PyTorch, zero `vllm-tpu` at runtime (`MAPS_LIBPYTHON=0`).**

---

## Architecture (M1)

| Component | File(s) | Responsibility |
| --- | --- | --- |
| **PJRT Runtime & Cache** | `pjrt_runtime.hpp`, `pjrt_runtime.cpp` | `dlopen(libtpu.so)` -> `GetPjrtApi()`, pure C++ `CompileOptionsProto` wire encoder, thread-safe LRU executable cache (`PJRT_LoadedExecutable*`), HBM buffer H2D/D2H, and built-in **Host Mock PJRT** (`CLOUDTPU_MOCK=1`) executing textual StableHLO IR on CPU for offline testing |
| **StableHLO Emitter** | `stablehlo_emitter.hpp`, `stablehlo_emitter.cpp` | Zero-dependency C++17 textual StableHLO MLIR emitter (`format="mlir"`) + deterministic structural cache key for `MUL_MAT` (`F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`), `RMS_NORM`, `ROPE` (Normal & NeoX + YaRN), `SILU`, `SWIGLU`, `SOFT_MAX` (mask/scale/ALiBi), `GET_ROWS`, `ADD`, `MUL`, `CPY`, `RESHAPE` |
| **GGML Backend** | `ggml-cloudtpu.h`, `ggml-cloudtpu.cpp` | HBM-resident `ggml_backend_buffer` slot manager (lossless split `(qs, d)` upload for `Q8_0`/`Q4_0`, `ggml_gallocr` overlap eviction), greedy contiguous subgraph fuser (`CLOUDTPU_FUSE_GRAPHS=1`), and `ggml_backend_sched` fallback |
| **PJRT Header** | `third_party/openxla/pjrt_c_api.h` | Pinned Apache-2.0 OpenXLA `pjrt_c_api.h` (`0bfeeb476952fc0ae13dd38e27a9053cb1baa3ce`, ABI `0.116`, `struct_size=1144`) |

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
  -o "MUL_MAT,RMS_NORM,ROPE,SOFT_MAX,GET_ROWS,ADD,MUL,CPY,UNARY,GLU"
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
