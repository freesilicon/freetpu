---
name: freetpu
description: >-
  Write, verify, and benchmark custom TPU kernels on Google Pixel Tensor G5
  (buenos_a0) via libopenbuenos/VLIW ISA tools and on Cloud TPU (v5e-1 / v6e-1)
  via Pallas/Mosaic and headless google-colab-cli. Use when asked to author,
  fuse, port, debug, or benchmark Edge TPU or Cloud TPU kernels, run offline
  host-mock parity checks, or execute zero-waste Colab TPU runs.
---

# FreeTPU Kernel & Benchmark Skill

Write, verify, and benchmark custom TPU kernels across **Pixel Tensor G5 (`buenos_a0`)** and **Cloud TPU (`v5e-1` / `v6e-1`)** with zero proprietary vendor binaries in source control.

Load [references/silicon-cheatsheet.md](references/silicon-cheatsheet.md) when choosing tile shapes, debugging VMEM/SRAM overflows, or working around Tensor G5 `INT4` / `W8A16` / DVFS errata.

## Pre-Flight Rules

1. **Clean-room boundary**: Never commit or redistribute proprietary `.so`, `.ko`, `.tflite`, `.dgc0`, or firmware blobs (e.g., `dive_interp_rv64.bin`). Physical Tensor G5 runs extract `ext[1]` locally on-device from the user's own `.dgc0` via `OPENBUENOS_DIVE_INTERP_PATH` or `ob_gemm_extract_dive_interp_from_dgc0`.
2. **Route by target**:
   - **Mobile / bare-metal Edge TPU (`buenos_a0`)**: Use Workflow 1 (`libopenbuenos`, `tools/emit_custom_gemm.py`, `tools/spline_0x640.py`, `tpu_baremetal_runner`).
   - **Datacenter Cloud TPU (`v5e-1` / `v6e-1`)**: Use Workflow 2 (`google-colab-cli` / `tools/colab_tpu_bench.py`) + Workflow 3 (Pallas/Mosaic).
3. **Parity before timing**: Always check numerical parity against NumPy `cpu_ref` (and XLA on Cloud TPU) before reporting latencies or TFLOP/s.

---

## Workflow 1: Pixel Tensor G5 (`buenos_a0`) Bare-Metal & Host Mock

Use this workflow for zero-Google-`.so` C/C++ kernels, closed-form `(M, K, N)` VLIW GEMMs, and `0x640` minimax spline fusion.

1. **Verify host mock runtime & C11/C++17 headers offline** (macOS/Linux, no phone required):
   ```bash
   cmake -S libopenbuenos -B libopenbuenos/build/cmake-host -DCMAKE_BUILD_TYPE=Release
   cmake --build libopenbuenos/build/cmake-host -j
   ctest --test-dir libopenbuenos/build/cmake-host --output-on-failure
   ```
   Set `ob_config_t::mock_mode = 1` (or `OPENBUENOS_MOCK=1`) in host binaries to exercise the DMA-BUF slab pool, LRU graph cache, 16-credit VII ring, and CPU reference parity without `/dev/edgetpu-soc`.

2. **Synthesize & verify VLIW GEMM + `0x640` minimax splines offline**:
   ```bash
   python3 tools/emit_custom_gemm.py --self-test
   python3 tools/spline_0x640.py --self-test --no-emit
   python3 skills/freetpu/templates/buenos_gemm_probe.py --self-test
   ```
   - `tools/emit_custom_gemm.py` synthesizes `chunk[0..6]` and packed `ext[2]` (`int32` bias + `[K/4, N_blk, 4]` `uint8` `w + 128`) using IEEE-754 **FP32 Round-Half-to-Even (RNE)** (`np.rint`), matching `buenos_a0` hardware rather than fixed-point RHU.
   - `tools/spline_0x640.py` synthesizes 208-byte `0x640` (`LoadCoefficientTables`) 8-segment degree-4 Remez minimax splines (`SiLU`, `GELU`, `exp`) with zero-overhead Mode-A fusion.
   - To emit a runnable `.dgc0` + `.in.bin` + `.ref.bin` bundle locally, pass a user-extracted shell via `--shell /path/to/local_shell.dgc0` (or `FREETPU_SHELL_DGC0`).

3. **Cross-build and prove on physical Tensor G5 silicon** (unlocked Android ARM64 device):
   ```bash
   make -C libopenbuenos verify-header android verify-android
   make -C runtime verify-baremetal
   ```
   Push `tpu_baremetal_runner` and test artifacts via `adb push` to `/data/local/tmp/`, run with `--ref`, and verify both bit-exact parity (`max_abs_err == 0`) and clean-room ELF mapping (`maps_has_google_tpu_libs == false`):
   ```bash
   adb shell "/data/local/tmp/tpu_baremetal_runner \
     --dgc0 /data/local/tmp/probe.dgc0 \
     --in /data/local/tmp/probe.in.bin \
     --ref /data/local/tmp/probe.ref.bin \
     --iters 10 --json"
   ```
   - **Errata guard**: Never pack 4-bit weights as nibbles for `0x980` (`LiteRT#12178`); unpack `Q4_0` at load time to `uint8` (`w + 8`) or use `W8A16 (BF16)` with per-channel `FP32` scales. See [references/silicon-cheatsheet.md](references/silicon-cheatsheet.md).

---

## Workflow 2: Headless Cloud TPU (`v5e-1` / `v6e-1`) via `google-colab-cli`

Use this workflow to run Pallas/Mosaic or JAX/PJRT benchmarks on Colab TPUs with zero idle Compute Unit (CU) waste.

1. **Check CU balance & isolate session state** (always pass global flags before the subcommand):
   ```bash
   colab --auth=adc --config /tmp/freetpu-colab.json usage
   ```
   - Abort if `Current balance` is below your run budget (e.g., `< 5.0` CUs).
   - Always pass `--config <unique-path>` so parallel workers never clobber `~/.config/colab-cli/sessions.json`.
   - Pass `--no-public-api` when targeting `/tun/m/*` tunnel auth on unallowlisted GCP quota projects.

2. **Run ephemeral probe with auto-stop and `v6e1 -> v5e1` fallback**:
   - Preferred wrapper (`tools/colab_tpu_bench.py`):
     ```bash
     # Offline check & command preview
     python3 tools/colab_tpu_bench.py --self-test
     python3 tools/colab_tpu_bench.py --dry-run --out-dir /tmp/freetpu-colab-dry

     # Live single-shot run (v6e1 with automatic v5e1 fallback on 412/400)
     python3 tools/colab_tpu_bench.py \
       --auth adc \
       --min-cu-balance 10 \
       --out-dir /tmp/freetpu-colab-proof \
       --script skills/freetpu/templates/tile_fused_gemm_silu.py
     ```
   - Direct CLI pattern (`colab run` provisions, runs, propagates exit code, and auto-stops the VM even on error):
     ```bash
     colab --auth=adc --config /tmp/freetpu-colab.json run \
       -s freetpu-probe --tpu V6E1 --timeout 300 \
       skills/freetpu/templates/tile_swiglu.py
     colab --auth=adc --config /tmp/freetpu-colab.json log \
       -s freetpu-probe -o /tmp/freetpu-colab-proof/proof.ipynb
     ```
   - For multi-step interactive sessions, prefer `colab exec` (keeps XLA/PJRT caches warm) over `colab ssh` (capped at 5 sessions / HTTP 429). Use `colab download` instead of `drive.mount`.

---

## Workflow 3: Authoring & Benchmarking Tile-Based TPU Kernels

Start from the templates in `templates/`—each supports `--self-test` offline on CPU (NumPy) and full `tpu_custom_call` execution on Cloud TPU:

| Target | Template | Purpose |
| --- | --- | --- |
| Cloud TPU (`v5e-1` / `v6e-1`) | [templates/tile_fused_gemm_silu.py](templates/tile_fused_gemm_silu.py) | Double-buffered `GEMM + SiLU` via `pl.pallas_call` + `pltpu.make_async_copy` vs XLA & NumPy `cpu_ref` |
| Cloud TPU (`v5e-1` / `v6e-1`) | [templates/tile_swiglu.py](templates/tile_swiglu.py) | Bit-exact `SwiGLU` in `FP32 -> BF16` + HBM streaming bandwidth (`GB/s`) probe |
| Pixel Tensor G5 (`buenos_a0`) | [templates/buenos_gemm_probe.py](templates/buenos_gemm_probe.py) | Offline INT8 FP32-RNE GEMM + `0x640` minimax spline synthesis & optional `.dgc0` emission |

### Authoring Rules

- **Cloud TPU (Pallas / Mosaic)**:
  1. Tile `M, N, K` in multiples of MXU/vector lanes (`(8, 128)` minimum; `(512, 512, 256)` to `(1024, 1024, 256)` for large `4096^3` GEMMs).
  2. Keep per-kernel scratch inside the **`16 MiB` VMEM** budget (`vmem_limit_bytes=16 * 1024 * 1024` via `pltpu.CompilerParams` or `pltpu.TPUCompilerParams`).
  3. Accumulate in `jnp.float32` (`preferred_element_type=jnp.float32`) and cast to `jnp.bfloat16` only on final VMEM -> HBM store.
  4. Overlap HBM DMA with MXU compute via `pltpu.emit_pipeline` or explicit 2-slot `pltpu.make_async_copy(..., pltpu.SemaphoreType.DMA((2,)))`.
  5. Confirm native Mosaic lowering (`"tpu_custom_call" in fn.lower(...).as_text()`) before benchmarking.
  6. Note the **`~115-135 us` sync PJRT dispatch floor**: fuse epilogues (SiLU, SwiGLU, RMSNorm) or pipeline async dispatches rather than chaining tiny kernels.

- **Tensor G5 (`buenos_a0`)**:
  1. Respect `768 KiB` NM SRAM and default `libopenbuenos` tile limits (`64 x 512 x 256`); align `K` to 4 (or 32 for `Q8_0`/`Q4_0` blocks) and `N` to 8/32.
  2. Match hardware **IEEE-754 FP32 RNE** (`np.rint`) when computing CPU reference tensors.
  3. Fuse activations (`SiLU`, `GELU`, `exp`) into `0x640` minimax splines or `0xa40` chained graphs to avoid LPDDR round-trips (`27.59 GB/s` ceiling).
  4. Benchmark with `PM=5` (`1,286.4 MHz`) + Firmware WakeLock (`perf_mode >= 1`), never default `PM=0` (`38.4 MHz`).

### Verification & Reporting Checklist

1. Run offline syntax & parity self-test locally:
   ```bash
   python3 tools/validate_skill.py --run-self-tests
   ```
2. On Cloud TPU, verify `"tpu_custom_call": true`, `max_abs_err` vs `cpu_ref`/XLA, warmup + `jax.block_until_ready()`, `median_us`, and `TFLOP/s` or `GB/s`, and save `<session>.ipynb` via `colab log -o`.
3. On Tensor G5, report `hw_us`, `wall_ms`, `max_abs_err == 0`, and `maps_has_google_tpu_libs == false`.
