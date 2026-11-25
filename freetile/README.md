# `freetile` — Zero-Dependency C++17 Portable Tile IR, Mosaic & Buenos Backends

`freetile/` (`namespace freetile::`) is a clean-room, zero-external-dependency C++17 tile programming IR and code generator for Cloud TPUs (`V5E`, `V6E`, `TPU7X`) and Edge Tensor G5 (`BUENOS_A0`).

## Design

- **Zero external dependencies:** Pure C++17 standard library only (`-Wall -Wextra -Wpedantic -Werror`). No LLVM, MLIR, Python, JAX, or libprotobuf dependencies.
- **Portable Tile IR (`include/freetile/tile_ir.hpp`):**
  - `ChipTarget` / `TargetProfile` (`V5E`, `V6E`, `TPU7X`, `BUENOS_A0`)
  - `TileGrid`, `SramTile`, `DmaPrefetch`, `QuantFormat` (`NONE`, `Q4_K`, `Q6_K`, `Q4_0`, `Q8_0`), `EpilogueActivation` (`NONE`, `RELU`, `RELU6`, `SILU`, `GELU`), `TileMatmul`, `TileKernel`
  - `TilePlannerConfig` & `TileScheduleHookFn` extension seam allowing external runtime/tuning layers to override `(bm, bn, bk)` schedules with zero fork.
- **Cloud TPU Mosaic Lowering (`include/freetile/mosaic_emitter.hpp`):**
  - Lowers `TileKernel` descriptors directly to `stable_mosaic.*` MLIR (`stable_mosaic.version = 17`) with RFC 4648 base64 encoding (`serialization_format = 1`) for `stablehlo.custom_call @tpu_custom_call`.
  - Integrated into `llama-cpp-cloudtpu` behind `CLOUDTPU_FREETILE_MOSAIC=1`, with automatic runtime fallback to pure StableHLO (`Q4_K_I4`, `Q6_K_SPLIT`, `Q4_0_I4`, `Q8_0_SPLIT`) if compilation or dialect version fails.
- **Tensor G5 (`BUENOS_A0`) VLIW / `DGC0` Lowering (`include/freetile/buenos_emitter.hpp`):**
  - Lowers `ChipTarget::BUENOS_A0` `TileKernel` descriptors (`Q8_0` general 81-shape multi-tile, 16/9-core `lt_1x*`, batched `lt_64x*`, weight-agnostic `W8A8_DYN_WEIGHTS`, 7-chunk per-tensor `W8A16_PT_1X` / `W8A16_PT_64X`, and zero-overhead Mode-A degree-4 Remez fused `SiLU`/`GELU` epilogues) directly to 7-chunk VLIW descriptors, DIVE VM IR (`ext[0]`), packed parameters (`ext[2]`), and 64-byte-aligned `DGC0` FlatBuffer executables.
  - Emits **11-chunk zero-DRAM in-SRAM chained SwiGLU** (`try_emit_buenos_swiglu` / `emit_buenos_swiglu`: `gate -> 0x640 SiLU * up [0x100 MUL] -> down`) with 100% bit-exact parity against `libopenbuenos` across all 6 silicon-proven shapes (`M=1`: `(576,512,576)`, `(576,1536,576)`, `(1536,512,1536)`, `(1536,1024,1536)`; `M=64`: `(576,512,576)`, `(1536,512,1536)`), plus an exact IEEE-754 RNE Horner simulator (`buenos_simulate_swiglu_exact`).
  - Extends 7-chunk per-tensor `W8A16(FP16)` (`[N/32, K/2, 32, 2]` `w^0x80` layout, `FP32` accumulator, zero intermediate `FP16` overflow hazard) to `K=576`/`N=576` (`SmolLM2-135M`, `M=1`) and batched `M=64` prefill (`W8A16_PT_64X`, `K,N in {512,576,1024,1536}` within the `768 KiB` Core-0 narrow SRAM ceiling), tagged `silicon_verified = false` until live `rango` verification; includes `buenos_should_prefer_w8a16(...)` encoding the selective mixed-precision policy (`lm_head`, `down_proj`, Layer 0–1, `Q6_K`, outlier crest guard, `>32768` FP16 prescale advisory).
  - Pure C++17 standard library only (zero dependency on `libopenbuenos`, LLVM, MLIR, or Python; `-fno-exceptions -fno-rtti` safe). Optional `test_buenos_parity` verifies bit-exact byte-for-byte parity against `libopenbuenos` when the sibling directory is present.

## Build & Test

```bash
cmake -S freetile -B freetile/build -DCMAKE_BUILD_TYPE=Release
cmake --build freetile/build
ctest --test-dir freetile/build --output-on-failure
```

To build `freetile` standalone without the optional `libopenbuenos` parity test:

```bash
cmake -S freetile -B freetile/build -DCMAKE_BUILD_TYPE=Release -DFREETILE_BUILD_BUENOS_PARITY_TEST=OFF
```

## Live `v5e-1` Silicon Verification (`MAPS_LIBPYTHON=0`, `LDD_LIBPYTHON=0`)

- `test-backend-ops -b CLOUDTPU -o MUL_MAT` (`CLOUDTPU_FREETILE_MOSAIC=1`): **`1361/1361` passed** (`mosaic_compiles=29`, `mosaic_fallbacks=0`).
- Fallback proof (`CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_FREETILE_MOSAIC_VERSION=999`): **`114/114` passed** (`mosaic_compiles=0`, `mosaic_fallbacks=9`).
- `Meta-Llama-3.1-8B-Instruct Q4_K_M` (`llama-bench -ngl 99 -fa 1`):
  - Pure StableHLO (`CLOUDTPU_FREETILE_MOSAIC=0`): `pp512 = 7041.18 tok/s`, `tg64 = 84.04 tok/s`
  - FreeTile Mosaic (`CLOUDTPU_FREETILE_MOSAIC=1`, `FUSE_ATTN=1`): `pp512 = 6932.82 tok/s` (`7047.48` at `FUSE_ATTN=0`), `tg64 = 80.65 tok/s` (`mosaic_compiles=1`, `mosaic_fallbacks=0`)
  - Isolated `4096x14336` `Q4_K` kernel (`16x` in-graph): `73.55 us` vs `105.06 us` raw XLA (`1.43x`); `14336x4096`: `74.19 us` vs `119.10 us` (`1.61x`).
