# Cloud TPU `v5e-1` Experimental Prototypes (Track B)

Clean-room experimental scripts and C++17 PJRT runtime prototypes validating Track B of [freetpu#3](https://github.com/saariuslystoned/freetpu/issues/3) on an ephemeral Google Colab `v5e-1` (`TPU v5 lite`, `15.748 GiB` HBM) runtime.

> **Scope & Clean-Room Boundary**: Colab is strictly an ephemeral single-chip R&D/benchmarking sandbox (no marketplace hosting or serving). This directory ships **zero proprietary binaries** (`.so`, `.bin`, headers). Build-time C++ compilation fetches OpenXLA's Apache-2.0 `pjrt_c_api.h` directly from upstream GitHub and dynamically loads the host VM's existing `libtpu.so` via `dlopen`.

---

## Files

| File | Track B Goal | Purpose |
| --- | --- | --- |
| `vfio_pci_survey.py` | **#4 VFIO/PCI & PJRT ABI Survey** | Read-only inspection of `/dev/vfio/*`, `tpu-info`, `jax.devices()`, dynamic `/sys/bus/pci/devices/*` (`BAR0`/`BAR4`, MSI-X, `vfio-pci`), `GetPjrtApi()` version & `struct_size`, and JAX Pallas/Mosaic TPU APIs |
| `raw_compute_bench.py` | **#0 Raw Compute & HBM BW Baseline** | Raw `jax.lax.dot_general` BF16/INT8 GEMM sweep (`1x64x32`..`16384^3`) and HBM streaming triad bandwidth (`a + s * b`) |
| `pallas_opt_bench.py` | **#3 Pallas GEMM+SiLU & FlashAttention** | Tile & `async_copy` double-buffering sweep on `4096^3` BF16 `GEMM + SiLU` + Triton-style `@tpu_flash_attn` causal attention vs dense XLA |
| `llama8b_decode_bench.py` | **#2 Llama-3-8B `W8A16`/`W4A16` Decode** | Fused VMEM-dequant Pallas `W8A16` / `W4A16` + `RMSNorm` + `SwiGLU` microbenchmarks and full 32-layer resident HBM `jax.lax.scan` synthetic decoder-layer step (`B=1`, `B=32`) |
| `export_stablehlo.py` | **#1 StableHLO IR Export & C++ Build** | Exports `.mlir` + `.mlirbc` + `CompileOptionsProto` into `~/.cache/freetpu-pjrt/ir`, verifies 26/26 `PJRT_Api` offsets against `libtpu.so`, fetches `pjrt_c_api.h`, and compiles checked-in `tpu_pjrt_bench.cpp` directly to `~/.cache/freetpu-pjrt/tpu_pjrt_bench` |
| `tpu_pjrt_bench.cpp` | **#1 Zero-Python C++17 PJRT Runtime** | Standalone CUDA-like `TpuRuntime` (`dlopen(libtpu.so)` -> `GetPjrtApi()`), `/proc/self/maps` zero-`libpython` check, `1x64x32` parity, sync/async/pipelined dispatch & `8192^3` TFLOP/s |

---

## Reproducing on a Cloud TPU VM (`v5e-1` / `v6e-1`)

Using `google-colab-cli` (`colab --auth=adc` with isolated `--config` under a private `0700` directory and `--tpu V6E1` or `V5E1`) or directly on a Cloud TPU VM:

```bash
mkdir -p -m 700 ~/.cache/freetpu-colab ~/.cache/freetpu-pjrt
mkdir -p -m 700 ~/.cache/freetpu-pjrt/ir

# Headless ephemeral single-script run (provisions -> runs -> auto-stops)
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json run --tpu V6E1 -- experimental/cloud-tpu/raw_compute_bench.py

# Or run all stages inside one persistent session (export .ipynb + always stop on exit)
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json new -s freetpu-tpu --tpu V5E1
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu -f experimental/cloud-tpu/vfio_pci_survey.py
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu -f experimental/cloud-tpu/raw_compute_bench.py
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu -f experimental/cloud-tpu/pallas_opt_bench.py
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu -f experimental/cloud-tpu/llama8b_decode_bench.py
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json upload -s freetpu-tpu experimental/cloud-tpu/tpu_pjrt_bench.cpp /content/tpu_pjrt_bench.cpp
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json upload -s freetpu-tpu experimental/cloud-tpu/export_stablehlo.py /content/export_stablehlo.py
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu "import runpy; runpy.run_path('/content/export_stablehlo.py', run_name='__main__')"
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json restart-kernel -s freetpu-tpu
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json exec -s freetpu-tpu "import os, subprocess; bin_p = os.path.expanduser('~/.cache/freetpu-pjrt/tpu_pjrt_bench'); print(subprocess.check_output(['ldd', bin_p], text=True)); print(subprocess.check_output([bin_p], text=True))"
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json log -s freetpu-tpu -o ~/.cache/freetpu-colab/freetpu-tpu-proof.ipynb
colab --auth=adc --config ~/.cache/freetpu-colab/sessions.json stop -s freetpu-tpu
```

> **Note on `v6e-1` (Trillium) availability**: An ephemeral `colab run --tpu V6E1` survey verified `GetPjrtApi() struct_size=1144, version=0.116` on a `V6E1` host (with `0000:00:05.0` occupied by the `0x1ae0:0x0042` gVNIC NIC rather than TPU, prompting dynamic `/sys/bus/pci/devices/*` scanning in `vfio_pci_survey.py`). Follow-up full-suite `V6E1` allocation requests encountered intermittent Colab `503 Service Unavailable` / `ReadTimeout` capacity limits, so full end-to-end benchmarks below are anchored on verified **`v5e-1`** silicon.

Or run Stage A / Stage B directly on a local Cloud TPU VM:

```bash
# Stage A (JAX kernel active on /dev/vfio/0)
python3 experimental/cloud-tpu/vfio_pci_survey.py
python3 experimental/cloud-tpu/raw_compute_bench.py
python3 experimental/cloud-tpu/pallas_opt_bench.py
python3 experimental/cloud-tpu/llama8b_decode_bench.py
python3 experimental/cloud-tpu/export_stablehlo.py

# Stage B (Fresh process / restarted kernel with /dev/vfio/0 released; zero Python)
ldd ~/.cache/freetpu-pjrt/tpu_pjrt_bench
~/.cache/freetpu-pjrt/tpu_pjrt_bench
```

Or build checked-in `tpu_pjrt_bench.cpp` standalone in a private `0700` directory once `.mlirbc` artifacts exist in `$HOME/.cache/freetpu-pjrt/ir`:

```bash
mkdir -p -m 700 ~/.cache/freetpu-pjrt
mkdir -p -m 700 ~/.cache/freetpu-pjrt/ir
curl -fsSL https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h \
  -o ~/.cache/freetpu-pjrt/ir/pjrt_c_api.h
g++ -O3 -std=c++17 -Wall -Wextra -I"$HOME/.cache/freetpu-pjrt/ir" \
  experimental/cloud-tpu/tpu_pjrt_bench.cpp -ldl -pthread -o ~/.cache/freetpu-pjrt/tpu_pjrt_bench
~/.cache/freetpu-pjrt/tpu_pjrt_bench
```

---

## Verified `v5e-1` Silicon Results (`libtpu 0.0.50` / `jax 0.11.2`)

### 1. PCI / VFIO & PJRT ABI Surface (`vfio_pci_survey.py`)
- **PCI Device**: `0000:00:05.0`, `vendor=0x1ae0`, `device=0x0063`, `subsystem=0x1ae0:0x00af`, `driver=vfio-pci`, `iommu_group=0`, `dma_mask_bits=32`, `128` MSI-X vectors.
- **MMIO BARs**: `BAR0 = 4096.00 MiB` (`0xd00000000..0xdffffffff`, `flags=0x14220c`), `BAR4 = 16.00 MiB` (`0xe00000000..0xe00ffffff`, `flags=0x14220c`).
- **On-Chip SRAM & Compute (`TpuInfo`)**: `128 MiB` VMEM, `1 MiB` SMEM, `4` MXUs, `15.748 GiB` (`16,909,336,064 B`) usable HBM, `820 GB/s` HBM BW, `197 TF/s` BF16, `394 TOPS` INT8, `788 TOPS` INT4.
- **PJRT C API**: `GetPjrtApi()` -> `struct_size=1144`, `version=0.116` (`libtpu.so` `601.67 MB`). All 26 required PJRT entrypoints verified inside `struct_size` and non-null.

### 2. Zero-Python C++17 PJRT Runtime (`tpu_pjrt_bench.cpp`)
- **Zero Python Proven**: `ldd` links only `libstdc++`, `libgcc_s`, `libc`, `libm`, `ld-linux`; runtime `/proc/self/maps` reports **`MAPS_LIBPYTHON=0`**.
- **Numeric Parity (`1x64x32` `bf16->f32`)**: **`max_abs_err = 0.000000e+00`** vs C++ host reference.
- **Dispatch Floor (`1x64x32`)**:
  - `sync_ready_event`: **`126.94 us`** median (`115.56 us` min)
  - `sync_device_complete`: **`124.52 us`** median (`116.43 us` min)
  - `sync_to_host`: **`231.16 us`** median (`205.57 us` min)
  - `async_enqueue_only`: **`28.86 us`** median (`20.68 us` min)
  - `pipelined_100x` async queue: **`32.00 us` amortized/launch** (`3.97x` lower than per-op sync)
- **Raw Compute (`sync_ready_event`)**:
  - `1024^3` (`bf16->f32`): **`141.99 us`** (`15.12` / `15.93 TFLOP/s`)
  - `4096^3` (`bf16->bf16`): **`949.31 us`** (`144.78` / `149.93 TFLOP/s`)
  - `8192^3` (`bf16->bf16`): **`6,145.94 us`** (**`178.90 TFLOP/s`** median / **`180.36 TFLOP/s`** peak = **`91.6%`** of 197 TF/s ceiling)

### 3. Pallas/Mosaic Optimization & FlashAttention (`pallas_opt_bench.py`)
- **`4096^3` BF16 `GEMM + SiLU`**: `pltpu.make_async_copy` double-buffered `1024x1024x256` reached **`135.50 TFLOP/s` median (`138.29 TFLOP/s` peak) = `93.99%` of XLA** (`144.16 TFLOP/s`), `max_abs_err=1.22e-4`, `tpu_custom_call=True`.
- **Causal FlashAttention (`@tpu_flash_attn`, `bs_512_1024_256`, `tpu_custom_call=True`)**:
  - `B=2, H=32, Q=K=2048, D=128`: **`1,386.44 us` (`49.57 TFLOP/s`)** vs dense XLA `4,913.38 us` (`13.99 TFLOP/s`) — **`3.54x` speedup**.
  - `B=1, H=32, Q=K=4096, D=128`: **`2,092.06 us` (`65.70 TFLOP/s` median / `66.65` peak)** vs dense XLA `9,436.90 us` (`14.56 TFLOP/s`) — **`4.51x` speedup** (`max_abs_err=0.015625`).

### 4. Resident 32-Layer Llama-3-8B Synthetic Decoder-Layer Steps (`llama8b_decode_bench.py`)
Exact Llama-3-8B hidden/FFN/GQA geometry (`d=4096, 32L, 32Q/8KV, head_dim=128, ff=14336`) resident in HBM via `jax.lax.scan`, repeatedly stepping synthetic activations across a fixed 256-entry synthetic KV cache and stopping at the final hidden state (omitting RoPE, LM-head vocabulary projection, and token sampling; throughput below is `batch / step_time` synthetic decoder-layer `steps/s`):
- **Single-Projection `W4A16` (`int32_split8_3d`, `max_abs_err = 0.0..0.0078`)**: `QKV 1x4096->6144` **`154.93 us`** (`2.02x` vs XLA), `Gate+Up 1x4096->28672` **`389.37 us`** (`3.24x` vs XLA `1,260.84 us`), `Down 1x14336->4096` **`243.66 us`** (`1.79x` vs XLA).
- **`W8A16` 32-Layer Resident (`6.98 GB` weights, `step0_max_abs_err=0.047` @ `B=1`)**:
  - `B=1`: XLA **`10.254 ms/step` (`97.53 steps/s`, `684.27 GB/s` = `83.55%` HBM BW)**; Pallas `17.840 ms/step` (`56.05 steps/s`).
  - `B=32`: XLA **`20.367 ms/step` (`1,571.15 steps/s`, `395.55 GB/s`)**; Pallas `35.045 ms/step` (`913.12 steps/s`).
- **`W4A16` 32-Layer Resident (`3.49 GB` packed `int32` weights)**:
  - `B=1`: Pallas **`15.529 ms/step` (`64.39 steps/s`, `227.09 GB/s`)** vs XLA `47.374 ms/step` (`21.11 steps/s`) — **`3.05x` speedup**.
  - `B=32`: Pallas **`27.416 ms/step` (`1,167.20 steps/s`, `166.57 GB/s`)** vs XLA `63.876 ms/step` (`500.97 steps/s`) — **`2.33x` speedup**.
  - *(Note: `W4A16` synthetic 32-layer `step0_max_abs_err` vs XLA was `16.0` @ `B=1` / `8.0` @ `B=32`: `alloc_stacked_w4_weights()` multiplies asymmetric `[0..15] - 8` random nibbles (`-0.5` mean) by strictly positive per-output scales `~[0.5, 1.5] / (8 * sqrt(K))` — with `1 / sqrt(64)` residual scaling on `wo`/`wd` — causing synthetic hidden states to grow across 32 residual layers into large-exponent BF16 where 1–2 ULPs equal `8.0`–`16.0`; single-projection `W4A16` parity vs XLA is `0.0..0.0078`.)*

---

## Side-by-Side Comparison vs Phase 6 Pixel Tensor G5 (`buenos_a0`)

| Metric | Pixel Tensor G5 (`buenos_a0`, `PM=5 @ 1,286.4 MHz` + WakeLock) | Cloud TPU `v5e-1` (`TPU v5 lite`, live Colab proof) |
| --- | --- | --- |
| **Kernel Transport** | `/dev/edgetpu-soc` (`buenos.ko` ABI `1.4`, Litebuf VII, 16-credit ring) | `/dev/vfio/0` (`vfio-pci`, PCI `0x1ae0:0x0063`, `BAR0=4 GiB`, `128` MSI-X) |
| **Userspace Vendor `.so`** | **`0 B`** (`libopenbuenos` C11/C++17; `DT_NEEDED=[libdl,libm,libc]`) | **`601.67 MB`** (`libtpu.so` `0.0.50`, PJRT `0.116`) — Zero-Python C++17 verified (`MAPS_LIBPYTHON=0`) |
| **Memory & Peak BW** | Unified LPDDR (`27.59 GB/s` measured @ `1x2048x2048`) + `768 KiB` NM SRAM | `15.748 GiB` HBM2e (**`655.55 GB/s`** triad / **`684.27 GB/s`** `W8A16` 32L) + `128 MiB` VMEM |
| **Small-Op Floor** | **`60.8–72.8 us`** chained `0xa40` fusion (`152 us` `1x2048x2048`) | **`124.5–126.9 us`** sync PJRT floor; **`32.0 us`** amortized when async-pipelined (`100x`) |
| **Peak Matrix Compute** | `983.6 INT8 GOPS` @ `177.2 mW` (**`5.55 TOPS/W`**) | **`180.36 TFLOP/s`** BF16 (C++ PJRT) / **`350.69 TOPS`** INT8 |
| **End-to-End LLM vs Synthetic Decoder Steps** | `Qwen2.5-1.5B-Q8_0` (end-to-end): `pp512 = 105.87 tok/s`, `tg64 = 5.85 tok/s` (`-21.5% mJ/tok` vs CPU) | `Llama-3-8B` (32L synthetic decoder steps/s, excludes RoPE/LM head/sampling): `W8A16` **`97.53 steps/s`** (`B=1`) / **`1,571.15 steps/s`** (`B=32`); `W4A16` Pallas: **`64.39 steps/s`** (`B=1`) / **`1,167.20 steps/s`** (`B=32`) |
