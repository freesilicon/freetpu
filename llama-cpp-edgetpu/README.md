# `ggml-edgetpu` — `llama.cpp` Backend for Pixel Tensor G5 (`buenos_a0`)

Clean-room GGML `ACCEL` backend (`EDGETPU`, device `"Pixel Tensor G5 (buenos_a0)"`) wiring upstream `ggml-org/llama.cpp` directly to `libopenbuenos` with zero Google/vendor `.so` dependencies.

---

## Included Files

- `ggml-edgetpu.h` — Public GGML EdgeTPU backend header (`ggml/include/ggml-edgetpu.h`)
- `ggml-edgetpu.cpp` — GGML `EDGETPU` backend, 64-byte aligned host-accessible buffer type, weight repacking, and `MUL_MAT` dispatch over `libopenbuenos` (`ggml/src/ggml-edgetpu/ggml-edgetpu.cpp`)
- `openbuenos_shim.c` — Standalone host mock shim when building without `libopenbuenos.a`
- `CMakeLists.txt` — Backend CMake module (`ggml/src/ggml-edgetpu/CMakeLists.txt`)
- `patches/llama-cpp-b12608-edgetpu.patch` — Minimal 3-file integration patch against `ggml-org/llama.cpp` tag `b12608`
- `tests/test_edgetpu_llm_shapes.cpp` — LLM shape accuracy sweep (`Q8_0 x F32`, `Q8_0 x Q8_0`, `F32 x F32`) & hybrid `ggml_backend_sched` routing verification
- `build.sh` — Overlay + host/Android ARM64 build + ELF `DT_NEEDED` audit script

---

## Integration with Upstream `ggml-org/llama.cpp` (`b12608`)

1. Build `libopenbuenos.a` first:
   ```bash
   make -C ../libopenbuenos host android
   ```
2. Clone upstream `llama.cpp` at tag `b12608` and apply the `EDGETPU` backend:
   ```bash
   git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp upstream/llama.cpp-b12608
   ./build.sh overlay
   ```
   Or manually:
   ```bash
   cp ggml-edgetpu.h upstream/llama.cpp-b12608/ggml/include/
   mkdir -p upstream/llama.cpp-b12608/ggml/src/ggml-edgetpu
   cp ggml-edgetpu.cpp openbuenos_shim.c CMakeLists.txt upstream/llama.cpp-b12608/ggml/src/ggml-edgetpu/
   git -C upstream/llama.cpp-b12608 apply ../../patches/llama-cpp-b12608-edgetpu.patch
   ```
3. Build host or Android ARM64 binaries:
   ```bash
   ./build.sh host      # offline host build (defaults to OPENBUENOS_MOCK=1)
   ./build.sh android   # Android ARM64 NDK cross-build + zero-vendor-.so ELF check
   ```

---

## Runtime Environment Toggles

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENBUENOS_MOCK` | `0` on Android, `1` on host | `1` runs offline mock `libopenbuenos` without `/dev/edgetpu-soc` |
| `OPENBUENOS_DIVE_INTERP_PATH` | `/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin` | Path to user-extracted RV64 DIVE interpreter image (`ext[1]`) |
| `OPENBUENOS_VERBOSE` | `0` | `1` logs backend init, tile counts, cache hits, `hw_us` / `wall_us` |
| `OPENBUENOS_MIN_ELEMENTS` | `0` | Minimum `M * K * N * batch` work threshold; smaller ops fall back to CPU |
| `OPENBUENOS_PACKED_Q8` | `0` | `1` routes `Q8_0 x F32` through pre-packed per-row INT8 `ob_gemm_matmul_int8` |
| `GGML_DISABLE_EDGETPU` | unset | Set `1` to disable `EDGETPU` registration for pure CPU baselines |

See [docs/llama-cpp-edgetpu/README.md](../docs/llama-cpp-edgetpu/README.md) for architecture details and test outputs.
