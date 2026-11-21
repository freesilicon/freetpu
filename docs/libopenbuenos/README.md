# `libopenbuenos` — Zero-Google-`.so` Bare-Metal Tensor G5 TPU Runtime

Reusable C/C++ runtime library (`libopenbuenos.a`) driving Pixel 10 Pro Fold (`rango`, Tensor G5 / `buenos.ko`) directly over `/dev/edgetpu-soc` (Litebuf VII) and `/dev/dma_heap/system` (`EDGETPU_MAP_DMABUF`), built specifically to power the `llama.cpp` (`ggml-edgetpu`) backend and standalone native TPU runners.

- **Public C11/C++17 ABI header**: [`libopenbuenos/include/openbuenos.h`](../../libopenbuenos/include/openbuenos.h)
- **Wire & kernel ABI inputs** (read-only):
  - [`docs/baremetal-vii/litebuf_vii_wire.h`](../baremetal-vii/litebuf_vii_wire.h)
  - [`docs/kernel-abi/edgetpu_buenos_uapi.h`](../kernel-abi/edgetpu_buenos_uapi.h)

---

## Architecture & Guarantees

1. **Zero Google / Vendor `.so` Dependencies**:
   - Compiled with `-std=c++17 -O2 -Wall -Wextra -Werror -fno-exceptions -fno-rtti`.
   - Android ARM64 PIE self-test (`build/android-arm64/openbuenos_selftest`) links only `[libdl.so, libc.so]` (`libdl.so` solely for static libunwind `dl_iterate_phdr`; zero `dlopen`/`dlsym`/`liblog`/vendor/Google libraries).
2. **DMA-BUF Slab/Pool Allocator (`ob_buffer_*`)**:
   - Allocates from `/dev/dma_heap/system` (`DMA_HEAP_IOCTL_ALLOC`), mmaps read/write, and maps 36-bit IOVAs via `EDGETPU_MAP_DMABUF`.
   - Power-of-two slab buckets (4 KiB..4 MiB, 4 KiB-aligned above) reuse mapped DMA-BUFs across token steps up to `ob_config_t::buffer_pool_max_bytes` (default 64 MiB), with explicit `ob_buffer_sync_to_device` / `ob_buffer_sync_to_cpu` cache syncs.
3. **Persistent Graph LRU Cache (`ob_graph_register_dgc0`)**:
   - Keyed by caller `cache_key` (if non-zero) or standalone SHA-256 over the `.dgc0` payload.
   - Cache hits skip DMA upload and `RegisterModel` ioctls completely (zero per-token graph registration cost). Entries are ref-counted across live `ob_graph_t*` handles and in-flight async submissions; LRU eviction only reaps unreferenced entries and cleanly issues `UnregisterModel`.
4. **16-Credit Async VII Execution Ring (`ob_graph_submit_async` / `ob_graph_wait`)**:
   - Supports up to `OB_MAX_CREDITS = 16` concurrent in-flight Litebuf VII commands (`EDGETPU_GET_CREDIT_SIZE = 16`), each backed by a pre-allocated 4 KiB inner-DIVE Execute DMA-BUF.
   - Enforces `buenos.ko`'s 32-bit `seq` constraint (`seq >> 32 == 0`) and demuxes out-of-order `EDGETPU_VII_LITEBUF_RESPONSE` completions over `eventfd` (`RESPDATA` + `FATAL_ERROR`).
   - Returns `OB_ERR_BUSY` when all credits are occupied.
5. **Buenos Weight Packing & Tiled LLM GEMM (`ob_gemm_*`)**:
   - `ob_gemm_pack_weights_int8` emits exact `ext[2]` layout: `int32[n_pad]` little-endian bias + `uint8[k_pad/4, n_pad, 4]` weights (`(w + 128) & 0xff`, padded channels `n >= N` raw `0x00`, `n_pad = ceil(N/8)*8`). Verified 320/320 byte-identical to `work/device-runner/fc.dgc0` `ext[2]`.
   - `ob_gemm_register_emitter(ctx, fn, user_data)` snaps in a clean-room `(M, K, N)` `.dgc0` emitter (e.g. from `gemm-multitile`) with configurable tile ceilings (`ob_gemm_set_tile_limits`, default `64x512x256`).
   - `ob_gemm_matmul_int8` and `ob_gemm_matmul_f32_q8_0` automatically decompose arbitrary LLM shapes (`M=1..64, K=2048..4096, N=2048..11008`) into tile dispatches, with exact CPU reference fallback when `mock_mode=1` or no emitter is attached.

---

## Build & Verification Commands

### Make (builds both macOS host and Android ARM64 NDK r26 artifacts)

```bash
make -C libopenbuenos all verify
```

Produced artifacts:
- Host static lib: `libopenbuenos/build/host/libopenbuenos.a`
- Host unit tests: `libopenbuenos/build/host/openbuenos_host_test`
- Android ARM64 static lib: `libopenbuenos/build/android-arm64/libopenbuenos.a`
- Android ARM64 PIE self-test: `libopenbuenos/build/android-arm64/openbuenos_selftest`

### CMake (Host & Android ARM64 NDK r26)

```bash
CMAKE=/opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake
NDK_ROOT=/opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125
SRC=libopenbuenos

# macOS host
"$CMAKE" -S "$SRC" -B "$SRC/build/cmake-host" -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$SRC/build/cmake-host" -j
"$SRC/build/cmake-host/openbuenos_host_test"

# Android ARM64 (API 34)
"$CMAKE" -S "$SRC" -B "$SRC/build/cmake-android-arm64" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$SRC/build/cmake-android-arm64" -j
```

---

## Integrating with `llama.cpp` (`ggml-edgetpu`)

1. **CMake wiring** in `ggml/src/ggml-edgetpu/CMakeLists.txt`:
   ```cmake
   set(OPENBUENOS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
   add_subdirectory(libopenbuenos
                    ${CMAKE_BINARY_DIR}/libopenbuenos)
   target_link_libraries(ggml-edgetpu PRIVATE openbuenos::openbuenos)
   ```
   Or link prebuilt `libopenbuenos/build/android-arm64/libopenbuenos.a` with `-I.../libopenbuenos/include`.

2. **Backend lifecycle**:
   ```c
   ob_config_t cfg = ob_config_default();
   cfg.mock_mode = use_mock ? 1 : 0;   // 1 for host tests, 0 on rango
   cfg.max_inflight = 16;              // full Buenos credit_size
   cfg.graph_cache_capacity = 256;     // keep layer tile graphs warm across decode tokens
   cfg.buffer_pool_max_bytes = 128u * 1024u * 1024u;

   ob_context_t* ctx = NULL;
   ob_context_create(&cfg, &ctx);
   ```

3. **Attaching a clean-room `(M, K, N)` `.dgc0` emitter**:
   ```c
   ob_gemm_set_tile_limits(ctx, 64, 512, 256);
   ob_gemm_register_emitter(ctx, my_buenos_dgc0_emitter, my_emitter_ctx);
   ```
   When `my_buenos_dgc0_emitter` returns `OB_OK` with a `.dgc0` buffer, `libopenbuenos` caches its registered TPU handle by tile key so subsequent decode tokens skip `RegisterModel`. Returning `OB_ERR_UNSUPPORTED` cleanly falls back to CPU reference per tile.

4. **Dispatching `GGML_OP_MUL_MAT`**:
   - For `GGML_TYPE_Q8_0` weights (`ne0 = K`, `ne1 = N`, `K % 32 == 0`): call `ob_gemm_matmul_f32_q8_0(ctx, a_f32, b_q8_0, out_f32, M, K, N, &stats)`.
   - For direct `INT8` weights (`[N, K]`): call `ob_gemm_matmul_int8(...)`, or pre-pack with `ob_gemm_pack_weights_int8(...)`.
   - For custom `.dgc0` pipelines: register once via `ob_graph_register_dgc0(ctx, blob, len, cache_key, &g)` and pipeline up to 16 tiles via `ob_graph_submit_async` + `ob_graph_wait` / `ob_graph_wait_all`.

---

## Verified Test Summary (Offline Host + NDK r26 Cross-Build)

- **Header checks**: `openbuenos.h` passes `-fsyntax-only -Wall -Wextra -Werror` under host `clang` (C11/C++17) and NDK r26 `aarch64-linux-android34-clang{,++}` (C11/C++17), plus pure C11 link smoke test.
- **Host unit tests (`openbuenos_host_test`, 7/7 suites passed)**:
  - `test_buffer_pool`: slab allocation, 4K-aligned IOVA, pool hit reuse, capacity eviction, read/write/sync.
  - `test_dgc0_parser`: parsed real `fc.dgc0` (`16->16`, `ext[2]=320B`), `add.dgc0` (`2x4096->4096`, no params), `custom_gemm_microkernel.dgc0` (`16->32`, `ext[2]=640B`), plus corrupt container rejection.
  - `test_graph_cache`: SHA-256 and explicit `cache_key` hits, LRU eviction, handle lifecycle.
  - `test_async_16_credit_ring`: 16 concurrent submits, 17th `OB_ERR_BUSY`, out-of-order wait on slot #7, slot reuse, `ob_graph_wait_all` drain.
  - `test_weight_packing_parity`: 320/320 byte-exact match vs `fc.dgc0` `ext[2]` + `N=20 -> n_pad=24` `0x00` channel padding parity.
  - `test_pluggable_emitter_tiling`: 4-tile dispatch via `ob_gemm_register_emitter`, 4/4 graph cache hits on repeat token, exact int32 output match.
  - `test_llm_size_matmuls`: `2048x2048` INT8 (exact 0 error, 32 tiles), `2048x2048` F32xQ8_0 (`max_abs_err=1.09e-06`, 32 tiles), `4096x11008` F32xQ8_0 (`max_abs_err=1.54e-06..1.90e-06`, 344 tiles).
- **Android ARM64 ELF audit (`llvm-readelf -d` / `llvm-nm -u`)**:
  - `openbuenos_selftest`: `ELF 64-bit LSB pie executable, ARM aarch64`, `DT_NEEDED = [libdl.so, libc.so]`, zero Google/vendor `.so`, zero `dlopen`/`dlsym`/`dlclose`/`dlerror`.
