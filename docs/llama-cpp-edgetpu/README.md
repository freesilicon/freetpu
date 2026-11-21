# `llama.cpp` + `ggml-edgetpu` Backend (`libopenbuenos` / Tensor G5 `buenos_a0`)

100% offline cross-build and native GGML backend (`EDGETPU`, device `"Pixel Tensor G5 (buenos_a0)"`) wiring `llama.cpp` directly to our zero-Google-`.so` bare-metal Tensor G5 runtime (`libopenbuenos`).

---

## 1. Pinned Upstream & Linked Runtime Provenance

| Artifact | Version / Path | SHA-256 |
| --- | --- | --- |
| Upstream `ggml-org/llama.cpp` tarball | Tag `b12608` (`llama-cpp-edgetpu/upstream/llama.cpp-b12608.tar.gz`) | `585b34a421d4ff2f3bedf905725e0d53ec1b1b364f6316110e95c1f6bfdd117e` |
| `openbuenos.h` | `libopenbuenos/include/openbuenos.h` (ABI `0.1.0`) | `81561e98552e573fda515ea362795b2ee024361f601ec4271096cb09c521f04b` |
| `libopenbuenos.a` (macOS host) | `libopenbuenos/build/host/libopenbuenos.a` | `a488c33a595b77ff7b36834b449168b5b89163deedc21b8d13fd43a190a5bce2` |
| `libopenbuenos.a` (Android ARM64) | `libopenbuenos/build/android-arm64/libopenbuenos.a` | `6eeb56ab013ad95224a9045fd9298366f3141f496a1c7b1954bcf0cf98fd28f5` |
| Android NDK Toolchain | NDK r26 (`26.1.10909125`, `aarch64-linux-android34`, `c++_static`) | — |
| CMake / Ninja | Android SDK CMake `3.22.1` + Ninja | — |

---

## 2. `ggml-edgetpu` Backend Design

### Files Added / Modified in `upstream/llama.cpp-b12608/`

- [`ggml/include/ggml-edgetpu.h`](../../llama-cpp-edgetpu/ggml-edgetpu.h): Public C API (`ggml_backend_edgetpu_init`, `ggml_backend_is_edgetpu`, `ggml_backend_edgetpu_is_mock`, `ggml_backend_edgetpu_buffer_type`, `ggml_backend_edgetpu_reg`).
- [`ggml/src/ggml-edgetpu/ggml-edgetpu.cpp`](../../llama-cpp-edgetpu/ggml-edgetpu.cpp): Full GGML `ACCEL` backend (`EDGETPU`, device description `"Pixel Tensor G5 (buenos_a0)"`), custom 64-byte aligned buffer type with weight repacking, and `MUL_MAT` execution over `libopenbuenos`.
- [`ggml/src/ggml-edgetpu/CMakeLists.txt`](../../llama-cpp-edgetpu/CMakeLists.txt): Auto-detects `-DOPENBUENOS_DIR` (or sibling `libopenbuenos/`) and statically links `build/android-arm64/libopenbuenos.a` (Android) or `build/host/libopenbuenos.a` (host), with a standalone mock shim fallback if built without `libopenbuenos.a`.
- [`ggml/CMakeLists.txt`](../../llama-cpp-edgetpu/patches/llama-cpp-b12608-edgetpu.patch), [`ggml/src/CMakeLists.txt`](../../llama-cpp-edgetpu/patches/llama-cpp-b12608-edgetpu.patch), [`ggml/src/ggml-backend-reg.cpp`](../../llama-cpp-edgetpu/patches/llama-cpp-b12608-edgetpu.patch): `-DGGML_EDGETPU=ON` option and `ggml_backend_edgetpu_reg()` registration (honoring `GGML_DISABLE_EDGETPU=1` for A/B CPU baselines).

### Op Support & Hybrid Scheduler Split

- **Claimed (`supports_op == true`)**:
  - `GGML_OP_MUL_MAT` (`dst == F32`, contiguous non-permuted inputs, default precision):
    - `Q8_0 x F32` (`K % 32 == 0`) -> `ob_gemm_matmul_f32_q8_0` (or pre-packed buenos `ob_gemm_matmul_int8` when `OPENBUENOS_PACKED_Q8=1`)
    - `Q8_0 x Q8_0` (`K % 32 == 0`) -> dequantize activations + `ob_gemm_matmul_f32_q8_0`
    - `F32 x F32` (`K % 4 == 0`, `N >= 8`) -> dynamic per-row symmetric INT8 quantization -> `ob_gemm_matmul_int8` -> F32 dequantize
  - No-op view/sentinel ops (`GGML_OP_NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`).
- **Routed automatically to ARM64 CPU (`supports_op == false`)**:
  - `RMS_NORM`, `ROPE`, `SOFT_MAX`, `MUL_MAT_ID`, `GET_ROWS`, `ADD`, `MUL`, `SILU`, non-contiguous/permuted views, and explicit `GGML_PREC_F32`/`GGML_PREC_BF16` precision requests.
- **Buffer Type (`ggml_backend_edgetpu_buffer_type`)**:
  - 64-byte aligned host-accessible primary storage (`is_host == true`) so ARM64 CPU glue ops read model tensors with zero copy.
  - At `set_tensor` on `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` `Q8_0` tensors (or lazily on cache miss), repacks weights into Tensor G5 `buenos_a0` `ext[2]` layout (`int32[n_pad] + uint8[K/4, n_pad, 4]` encoded `(w + 128) & 0xff`) via `ob_gemm_pack_weights_int8` plus per-row scales.

### Runtime Environment Toggles

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENBUENOS_MOCK` | `0` on Android, `1` on host | `1` runs offline/mock `libopenbuenos` without `/dev/edgetpu-soc` |
| `OPENBUENOS_VERBOSE` | `0` | `1` logs backend init, weight repacks, per-op `(M, K, N)`, tiles, cache hits, `hw_us` / `wall_us` |
| `OPENBUENOS_MIN_ELEMENTS` | `0` | Minimum `M * K * N * batch` work threshold; smaller `MUL_MAT` ops fall back to CPU |
| `OPENBUENOS_PACKED_Q8` | `0` | `1` routes `Q8_0 x F32` through the pre-packed per-row INT8 `ob_gemm_matmul_int8` path |
| `GGML_DISABLE_EDGETPU` | unset | Set (`GGML_DISABLE_EDGETPU=1`) to disable `EDGETPU` registration for pure ARM64 CPU baselines |

---

## 3. Build Commands

One-command helper: [`llama-cpp-edgetpu/build.sh`](../../llama-cpp-edgetpu/build.sh) (`./build.sh host`, `./build.sh android`, or `./build.sh all`).

### macOS Host Build (`build-host`)

```bash
CMAKE=/opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake
NINJA=/opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/ninja
ROOT=llama-cpp-edgetpu
OB=libopenbuenos

"$CMAKE" -S "$ROOT/upstream/llama.cpp-b12608" -B "$ROOT/build-host" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_EDGETPU=ON \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_OPENSSL=OFF \
  -DOPENBUENOS_DIR="$OB"

"$CMAKE" --build "$ROOT/build-host" --target test-backend-ops llama-cli llama-bench -j"$(sysctl -n hw.ncpu)"
```

### Android ARM64 Cross-Build (`build-android-arm64`)

```bash
CMAKE=/opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake
NINJA=/opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/ninja
NDK=/opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125
ROOT=llama-cpp-edgetpu
OB=libopenbuenos

"$CMAKE" -S "$ROOT/upstream/llama.cpp-b12608" -B "$ROOT/build-android-arm64" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_EDGETPU=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16+i8mm" \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_OPENSSL=OFF \
  -DOPENBUENOS_DIR="$OB"

"$CMAKE" --build "$ROOT/build-android-arm64" --target test-backend-ops llama-cli llama-bench -j"$(sysctl -n hw.ncpu)"
```

---

## 4. Verification Results

### 4.1 Host Tests (Linked Directly Against `libopenbuenos/build/host/libopenbuenos.a`)

1. **Upstream `test-backend-ops -b EDGETPU -o MUL_MAT`**:
   ```text
   OPENBUENOS_MOCK=1 ./build-host/bin/test-backend-ops -b EDGETPU -o MUL_MAT
   Backend 1/2: EDGETPU
     Device description: Pixel Tensor G5 (buenos_a0)
     223/223 tests passed
     Backend EDGETPU: OK
   ```
2. **Pre-packed `Q8_0` INT8 mode (`OPENBUENOS_PACKED_Q8=1`)**:
   ```text
   OPENBUENOS_MOCK=1 OPENBUENOS_PACKED_Q8=1 ./build-host/bin/test-backend-ops -b EDGETPU -o MUL_MAT -p "type_a=q8_0,type_b=f32"
   Backend 1/2: EDGETPU
     104/104 tests passed
     Backend EDGETPU: OK
   ```
3. **Glue-op CPU fallback verification (`RMS_NORM, ROPE, SOFT_MAX, MUL_MAT_ID, GET_ROWS`)**:
   - All glue ops cleanly return `not supported [EDGETPU]` (`0/0` claimed).
4. **Device registration & disable toggle (`llama-cli --list-devices`)**:
   ```text
   $ OPENBUENOS_MOCK=1 ./build-host/bin/llama-cli --list-devices
   Available devices:
     EDGETPU: Pixel Tensor G5 (buenos_a0) (0 MiB, 0 MiB free)

   $ GGML_DISABLE_EDGETPU=1 ./build-host/bin/llama-cli --list-devices
   Available devices:
     (none)
   ```
5. **LLM Shape Sweep & Hybrid Scheduler Proof ([`test_edgetpu_llm_shapes.cpp`](../../llama-cpp-edgetpu/tests/test_edgetpu_llm_shapes.cpp))**:
   - **`Q8_0 x F32` (`M=1,4,16`, `K=64..2048`, `N=64..2048`)**: `12/12 PASS`, `nmse = 1.14e-05 .. 1.50e-05` (`<= 5e-4`), bit-exact `Q8_0` buffer readback confirmed.
   - **`Q8_0 x Q8_0` (`M=1,4`, `K=256..1024`, `N=256..1024`)**: `2/2 PASS`, `nmse = 1.56e-14 .. 2.35e-14`.
   - **`F32 x F32` (dynamic INT8 quantize -> `ob_gemm_matmul_int8` -> F32)**: `12/12 PASS`, `nmse = 2.81e-05 .. 3.61e-05` (`<= 5e-4`).
   - **Hybrid `ggml_backend_sched` routing**:
     ```text
     RMS_NORM -> CPU
     MUL_MAT  -> EDGETPU
     SOFT_MAX -> CPU
     Hybrid compute status=0, softmax row0 sum=1.000000 -> PASS
     ```

### 4.2 Android ARM64 Binaries & Zero-Google-`.so` Proof (`llvm-readelf -d`)

| Binary | Path | SHA-256 | `DT_NEEDED` | Linked `libopenbuenos` Symbols |
| --- | --- | --- | --- | --- |
| `llama-cli` | `build-android-arm64/bin/llama-cli` | `67851b8458979ce78e61b5fcd5c7ca99d0dbb356174e89f9d4136586bb74db30` | `libdl.so`, `libm.so`, `libc.so` | `ob_context_create`, `ob_gemm_pack_weights_int8`, `ob_gemm_matmul_int8`, `ob_gemm_matmul_f32_q8_0`, `ggml_backend_edgetpu_reg` |
| `llama-bench` | `build-android-arm64/bin/llama-bench` | `20fa1319d5e992acabbb1c7c29110cbc9a64fbfdfdb554aa20a61b89555bb337` | `libdl.so`, `libm.so`, `libc.so` | `ob_context_create`, `ob_gemm_pack_weights_int8`, `ob_gemm_matmul_int8`, `ob_gemm_matmul_f32_q8_0`, `ggml_backend_edgetpu_reg` |
| `test-backend-ops` | `build-android-arm64/bin/test-backend-ops` | `d325ff9cf5d8a3677772d14ca123403e2a97979e22c4cb4da4b74894d3d01999` | `libdl.so`, `libm.so`, `libc.so` | `ob_context_create`, `ob_gemm_pack_weights_int8`, `ob_gemm_matmul_int8`, `ob_gemm_matmul_f32_q8_0`, `ggml_backend_edgetpu_reg` |

**Zero** Google/vendor `.so` dependencies (`libedgetpu*`, `libdarwinn*`, `liblitert*`, `libc++_shared.so`, `libomp.so` are all absent).

---

## 5. Ready-to-Run `rango` Push & Benchmark Commands

*(Run by the USB-owning parent lane — never invoked by this offline lane.)*

### Optional: Strip Debug Symbols Before Push (Smaller Transfer)

```bash
STRIP=/opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip
ROOT=llama-cpp-edgetpu
mkdir -p "$ROOT/build-android-arm64/stripped"
for b in llama-cli llama-bench test-backend-ops; do
  "$STRIP" -o "$ROOT/build-android-arm64/stripped/$b" "$ROOT/build-android-arm64/bin/$b"
done
```

### Push Binaries & Model to `rango`

```bash
BIN_DIR=llama-cpp-edgetpu/build-android-arm64/bin
adb shell mkdir -p /data/local/tmp/llama-edgetpu
adb push "$BIN_DIR/test-backend-ops" "$BIN_DIR/llama-bench" "$BIN_DIR/llama-cli" /data/local/tmp/llama-edgetpu/
adb shell chmod +x /data/local/tmp/llama-edgetpu/test-backend-ops /data/local/tmp/llama-edgetpu/llama-bench /data/local/tmp/llama-edgetpu/llama-cli
# Push a Q8_0 GGUF model (e.g. SmolLM2-135M / Qwen2.5-0.5B / Llama-3.2-1B Q8_0):
# adb push /path/to/model-q8_0.gguf /data/local/tmp/llama-edgetpu/model-q8_0.gguf
```

### On-Device Verification (`rango`)

```bash
# 1. Verify device enumeration (mock vs live /dev/edgetpu-soc)
adb shell "OPENBUENOS_MOCK=1 /data/local/tmp/llama-edgetpu/llama-cli --list-devices"
adb shell "su -c '/data/local/tmp/llama-edgetpu/llama-cli --list-devices'"

# 2. On-device MUL_MAT accuracy check (offline mock first, then live /dev/edgetpu-soc as root)
adb shell "OPENBUENOS_MOCK=1 /data/local/tmp/llama-edgetpu/test-backend-ops -b EDGETPU -o MUL_MAT -p 'type_a=q8_0,type_b=f32'"
adb shell "su -c 'OPENBUENOS_VERBOSE=1 /data/local/tmp/llama-edgetpu/test-backend-ops -b EDGETPU -o MUL_MAT -p \"type_a=q8_0,type_b=f32,m=64,n=2,k=256\"'"
```

### Apples-to-Apples `Q8_0` Benchmark: ARM64 CPU Baseline vs `EDGETPU`

```bash
MODEL=/data/local/tmp/llama-edgetpu/model-q8_0.gguf

# Baseline: pure ARM64 CPU (EDGETPU disabled at registry init)
adb shell "GGML_DISABLE_EDGETPU=1 /data/local/tmp/llama-edgetpu/llama-bench -m $MODEL -p 128 -n 64 -t 4 -r 3"

# EDGETPU backend (live /dev/edgetpu-soc via libopenbuenos; tune OPENBUENOS_MIN_ELEMENTS as needed)
adb shell "su -c 'OPENBUENOS_MIN_ELEMENTS=65536 /data/local/tmp/llama-edgetpu/llama-bench -m $MODEL -p 128 -n 64 -t 4 -r 3'"

# Interactive smoke test with verbose EdgeTPU dispatch telemetry
adb shell "su -c 'OPENBUENOS_VERBOSE=1 OPENBUENOS_MIN_ELEMENTS=65536 /data/local/tmp/llama-edgetpu/llama-cli -m $MODEL -p \"Hello Tensor G5\" -n 32 --no-display-prompt'"
```
