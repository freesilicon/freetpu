# `ggml-cloudtpu` — Native C++17 `llama.cpp` Backend for Google Cloud TPUs

Clean-room Apache-2.0 GGML `GPU` backend (`CLOUDTPU`) targeting Google Cloud TPUs (`v5e`, `v6e`, `tpu7x`) directly over OpenXLA's C `PJRT_Api` (`dlopen(libtpu.so)` -> `GetPjrtApi()`).

**Zero Python, zero JAX, zero PyTorch, zero `vllm-tpu` at runtime (`MAPS_LIBPYTHON=0`).**

---

## Architecture (M1–M4)

| Component | File(s) | Responsibility |
| --- | --- | --- |
| **PJRT Runtime & Cache** | `pjrt_runtime.hpp`, `pjrt_runtime.cpp` | `dlopen(libtpu.so)` -> `GetPjrtApi()`, pure C++ `CompileOptionsProto` (typed `env_option_overrides` for opt-in TPU async collective overlap flags without overwriting `DebugOptions`/`XLA_FLAGS`) + `ExecutableBuildOptionsProto` + `DeviceAssignmentProto` wire encoder, thread-safe LRU executable cache (`PJRT_LoadedExecutable*`) with eviction epoch, single-chip & multi-chip replica-SPMD execution (`execute_spmd` with monotonic `launch_id` and `non_donatable_input_indices`), HBM buffer H2D/D2H with explicit untiled C-order `host_layout`, `/proc/self/maps` `libpython` audit, and multi-device **Host Mock PJRT** (`CLOUDTPU_MOCK=1`, `CLOUDTPU_MOCK_DEVICES=1..16`) executing StableHLO IR + `dynamic_update_slice` + cross-chip collectives |
| **StableHLO Emitter** | `stablehlo_emitter.hpp`, `stablehlo_emitter.cpp` | Zero-dependency C++17 textual StableHLO MLIR emitter (`format="mlir"`) + deterministic structural cache key for TPU-native K-major N-minor quantized `MUL_MAT` (`[B3, B2, NB, 16/32, N]` + split-X zero-concat Q4 dot + BF16 MXU), `FLASH_ATTN_EXT` (GQA + mask + ALiBi + softcap), `RMS_NORM`, `ROPE` (Normal & NeoX + YaRN + `rope_freqs`), `SILU`, `SWIGLU`, `SOFT_MAX`, `GET_ROWS` (last-axis gather on quantized tables), `SET_ROWS` (single-token `dynamic_update_slice` + `{tf.aliasing_output}` KV donation, multi-row `scatter`), `ADD`, `MUL`, `SCALE`, `CPY`, `RESHAPE`, `TRANSPOSE`, `SLICE`, and StableHLO v1.0 collectives (`ALL_REDUCE_SUM`, `ALL_GATHER`, `REDUCE_SCATTER_SUM`) |
| **GGML Backend** | `ggml-cloudtpu.h`, `ggml-cloudtpu.cpp` | HBM-resident multi-chip `ggml_backend_buffer` slot manager (K-major N-minor `(8,128)` tile-aligned quantized HBM upload & axis-aware SPMD sharding for `Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`, O(1) slot reuse fast-path, runtime-checked in-place KV buffer donation, sub-slice view patching), O(N) `ggml_gallocr` liveness precomputation, zero-MLIR-re-emission `SubgraphPlanCache`, extended fusion (`CLOUDTPU_MAX_FUSE_OPS=512`) with automatic Megatron-LM 1-collective FFN detection, async pipelining (`ggml_backend_cloudtpu_synchronize`), and recursive `VIEW`/`PERMUTE`/`TRANSPOSE`/`RESHAPE` lowering |
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

### M3 -> M4B -> M4C End-to-End GGUF Throughput (`TPU v5e-1`, `llama-bench -ngl 99 -fa 1`)

**M4C Decode Lift** adds six TPU-native kernel & runtime upgrades:
1. **TPU-native K-major N-minor quantized HBM layout (`[B3, B2, NB, 16/32, N]` + `[B3, B2, NB, N]`):** aligns quantized weights and scales directly with TPU `(8, 128)` HBM tiles (`0%` minor-dimension padding vs `4x`/`8x` bloat on `[N, NB, 32/16]`), eliminates weight reshapes/transposes, and broadcasts scales along axis `3` without HBM materialization.
2. **Zero-concatenate split-activation Q4 matmul (`CLOUDTPU_Q4_SPLIT_DOT=1`):** replaces weight `stablehlo.concatenate` with `dot(x_lo, w_lo) + dot(x_hi, w_hi)` on `[NB, 16, N]` nibble halves (`+78%` on `Llama-3.1-8B Q4_K_M` decode vs concatenated single-dot: `69.56` vs `39.34 tok/s`).
3. **Single-token `SET_ROWS` bounds-guarded `dynamic_update_slice` + in-place KV buffer donation (`CLOUDTPU_DONATE_KV=1`):** emits explicit `{tf.aliasing_output = idx : i32}` (with `alias_passthrough_params` disabled) and runtime exclusive-ownership donation checks (`non_donatable_input_indices` marks all non-aliased or shared inputs), while guarding out-of-range row indices to preserve scatter discard semantics.
4. **Llama-3 `ROPE` `n_dims == 0` probe fix:** keeps `rope_freqs.weight` resident in TPU HBM during `llama-model-loader` buffer placement, eliminating the 65 CPU/TPU graph splits per token on Llama-3.1-8B.
5. **BF16 MXU contraction (`CLOUDTPU_MATMUL_BF16=1`):** computes quantized affine `q * d + m` in F32 VPU precision and casts once to `BF16` for native MXU contraction (`114/114` `Q4_K`, `102/102` `Q8_0`, `63/63` `Q6_K`, `102/102` `Q4_0`, `287/287` `ROPE`, `65/65` `SET_ROWS`, `56/56` `GET_ROWS` passed on live `v5e-1`).
6. **Multi-chip async collective overlap (`CLOUDTPU_ASYNC_COLLECTIVES=1`, opt-in):** encodes 6 typed XLA TPU async collective overlap overrides (`5 bool_field` + `xla_enable_async_all_gather=kEnabled`) into `CompileOptionsProto.env_option_overrides` (preserving compiler defaults and `XLA_FLAGS`) plus monotonic `launch_id` when `P > 1`.

| Model (GGUF) | Quant | Size | M3 `pp512` | M4B `pp512` | **M4C `pp512`** | M3 `tg64` | M4B `tg64` | **M4C `tg64`** (`FUSE_ATTN=0` / `1`) | M4C vs M4B `tg64` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `16,113.17` | `28,132.27` | **`22,220.93`** | `39.17` | `192.06` | `190.19` / **`209.88`** | **+9.3%** |
| `Qwen2.5-1.5B-Instruct` | `Q8_0` | `1.76 GiB` | `19,820.42` | `28,371.30` | **`29,662.17`** | `42.24` | `202.96` | `206.00` / **`210.63`** | **+3.8%** |
| `Meta-Llama-3.1-8B-Instruct` | `Q4_K_M` | `4.58 GiB` | `4,684.60` | `4,726.56` | **`6,524.26`** | `33.25` | `38.74` | `69.56` / **`70.24`** | **+81.3% (1.81x)** |
| `Meta-Llama-3.1-8B-Instruct` | `Q8_0` | `7.95 GiB` | `4,207.50` | `4,237.41` | **`6,963.31`** | `35.20` | `53.06` | **`60.33`** / `57.66` | **+13.7% (>62% peak HBM BW)** |

*Both `Qwen2.5-1.5B` and `Meta-Llama-3.1-8B` pass live `v5e-1` `llama-cli` coherence (`17 * 23 = 391`) with verified `/proc/<pid>/maps` `MAPS_LIBPYTHON=0` and `LDD_LIBPYTHON=0`.*

### M4A Multi-Chip SPMD Live Silicon Proof (`v6e-8` Trillium, `MAPS_LIBPYTHON=0`)

When opted in via `CLOUDTPU_SPMD_CHIPS=auto` (or an explicit count such as `CLOUDTPU_SPMD_CHIPS=8`) with `P > 1` TPU chips, `ggml-cloudtpu` exposes a single unified `CLOUDTPU` device reporting per-chip bottleneck HBM (`min_chip_hbm`, with sharded weights charged `1/P` per chip and replicated KV/compute buffers charged `1x` per chip), compiles explicit replica-SPMD StableHLO v1.0 collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `replica_id` + `dynamic_slice`) with `DeviceAssignmentProto` (`num_replicas=P, num_partitions=1`), and automatically shards 2D weights by `1/P` across chips (`SHARD_K` for `ffn_down`/`token_embd`, `SHARD_N` for `Q`/`K`/`V`/`O`/`ffn_gate`/`ffn_up`/`output`, fusing Megatron-LM FFN blocks down to a single `all_reduce`). Leaving `CLOUDTPU_SPMD_CHIPS` unset (default `1`) preserves per-chip devices (`CLOUDTPU`, `CLOUDTPU1`, ...).

#### Live 8-Chip `v6e-8` Silicon Verification (`SPMD 8x TPU v6 lite`, `255,967 MB` HBM, `ldd libpython = 0`)
- **`test_cloudtpu_host` (`CLOUDTPU_MOCK=0`):** `ALL PASSED (0 failures)` — single-chip & P=1 StableHLO collective syntax/execution (`all_reduce`/`all_gather`/`reduce_scatter`) verified live on `libtpu.so` (`MAPS_LIBPYTHON=0`), plus 4-chip & 8-chip SPMD collective and Megatron FFN unit tests on the in-process Host Mock (live 8-chip SPMD execution is verified below via `test-backend-ops` and `llama-bench`/`llama-cli`).
- **`test-backend-ops -b CLOUDTPU` (`CLOUDTPU_SPMD_CHIPS=8`):** **`3,820 / 3,820` PASSED** across all 8 core suites:
  - `MUL_MAT`: `1,361/1,361` | `FLASH_ATTN_EXT`: `1,697/1,697` | `ROPE`: `287/287` | `ADD`: `153/153` | `MUL`: `140/140` | `SET_ROWS`: `65/65` | `RMS_NORM`: `61/61` | `GET_ROWS`: `56/56`.
- **End-to-End 8-Chip GGUF Inference (`llama-bench -ngl 99 -fa 1` & `llama-cli -c 4096`):**

| Model (GGUF) | Quant | Size | Chips (`P`) | `pp512` (tok/s) | `tg64` (tok/s) | `llama-cli` Coherence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `1` | `10,479.21` | `54.52` | Verified |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `8` | `5,212.55` | `12.13` | Verified |
| `Qwen3-32B` | `Q4_K_M` | `18.40 GiB` | `8` | `2,747.30` | `7.20` | Verified coherent reasoning |
| `Llama-3.3-70B-Instruct` | `Q4_K_M` | `39.59 GiB` | `8` | `502.72` | `2.23` | Verified (`17 * 23 = 391`) — exceeds single-chip 31.25 GiB HBM, runs sharded across 8 chips |

*Note: `v6e-8` M4A baseline proves end-to-end 8-chip 32B/70B correctness & zero-Python SPMD sharding; autoregressive `tg64` throughput is currently bottlenecked by out-of-place `SET_ROWS` (`stablehlo.scatter`), per-subgraph host sync / unfused `Q4_K` dequant, and CPU `rope_freqs.weight` graph splits on Llama-3 — targeted in the next decode acceleration slice (`dynamic_update_slice` buffer donation, whole-layer fusion, VMEM-fused dequant, and async collective overlap flags).*

| Target Topology (Phase 0 Baseline, PR `#15`) | Chips (`P`) | Total HBM | `Qwen3-32B` (`Q4_K_M` / `BF16`) per chip | `Llama-3.3-70B` (`Q4_K_M` / `Q8_0` / `BF16`) per chip | Verification Status |
| --- | ---: | ---: | --- | --- | --- |
| **`v6e-8`** (Trillium) | `8` | `256 GiB` (`32 GiB/chip`) | `~2.5 GiB` / `~8.1 GiB` | `~5.3 GiB` / `~9.4 GiB` / `~17.6 GiB` | **Verified live on `v6e-8` silicon** (`test_cloudtpu_host`, `3,820/3,820` `test-backend-ops`, `Qwen3-32B`, `Llama-3.3-70B` `39.59 GiB`, `MAPS_LIBPYTHON=0`) |
| **`tpu7x-standard-4t`** (Ironwood) | `4` (`8` logical) | `768 GiB` (`192 GiB/chip`) | `~5.0 GiB` / `~16.3 GiB` | `~10.6 GiB` / `~18.8 GiB` / `~35.3 GiB` | Verified on 4-chip & 8-chip Host Mock; live Ironwood silicon tracked in `#21` (`WAITING_FOR_QUOTA`) |

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

# 1-chip Host Mock verification against ggml-cpu:
CLOUDTPU_MOCK=1 ./llama-cpp-cloudtpu/build-host/bin/test-backend-ops -b CLOUDTPU \
  -o "MUL_MAT,RMS_NORM,ROPE,SOFT_MAX,GET_ROWS,ADD,MUL,CPY,SILU,SWIGLU"

# 8-chip v6e-8 SPMD Host Mock verification:
CLOUDTPU_MOCK=1 CLOUDTPU_MOCK_DEVICES=8 CLOUDTPU_SPMD_CHIPS=8 CLOUDTPU_SPMD_MIN_ROWS=8 \
  ./llama-cpp-cloudtpu/build-host/bin/test-backend-ops -b CLOUDTPU -o "MUL_MAT,ADD,MUL,RMS_NORM,SOFT_MAX"
```

---

## Runtime Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLOUDTPU_MOCK` | `0` on Linux, `1` on macOS/Windows | `1` runs in-process Host Mock PJRT executing emitted StableHLO IR |
| `CLOUDTPU_MOCK_DEVICES` | `1` | Virtual TPU chip count in Host Mock (`4` simulates `tpu7x-4t`, `8` simulates `v6e-8`) |
| `CLOUDTPU_SPMD_CHIPS` | `1` (per-chip devices) | Opt-in replica-SPMD Tensor Parallelism: `1` exposes per-chip devices (`CLOUDTPU`, `CLOUDTPU1`, ...); `auto` uses all discovered chips; `2..16` sets an explicit mesh size |
| `CLOUDTPU_SPMD_MIN_ROWS` | `256` | Minimum 2D weight matrix rows/cols required to shard across SPMD chips |
| `CLOUDTPU_MAX_FUSE_OPS` | `512` | Maximum GGML ops fused into a single StableHLO subgraph executable (`1..2048`) |
| `CLOUDTPU_FUSE_ATTN` | `0` | Set `1` to fuse `FLASH_ATTN_EXT` with following O-proj/FFN ops (`+10%` on `Qwen2.5-1.5B Q4_K_M`, `70.24 tok/s` on `Llama-3.1-8B Q4_K_M`) |
| `CLOUDTPU_DONATE_KV` | `1` | `1` enables in-place KV cache HBM donation via `dynamic_update_slice` + `{tf.aliasing_output}` |
| `CLOUDTPU_MATMUL_BF16` | `1` | `1` casts dequantized weights & activations to `BF16` for TPU MXU contraction (dense F32 weights stay F32) |
| `CLOUDTPU_Q4_SPLIT_DOT` | `1` | `1` uses zero-concatenate split-X Q4 `dot_general` (`+78%` on 8B `Q4_K_M`); `0` uses single concatenated `dot_general` |
| `CLOUDTPU_Q4K_BF16_SCALES` | `0` | Set `1` to store `Q4_K`/`Q6_K` scales in F16 (`0` = F32 scales, faster on `v5e-1`) |
| `CLOUDTPU_ASYNC_COLLECTIVES` | `0` | Opt-in (`1`) enables 6 typed XLA TPU async collective overlap overrides in `CompileOptionsProto.env_option_overrides` when `P > 1` |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | Auto-resolved `libtpu.so` | Explicit path to `libtpu.so` (or future clean-room `libopentpu.so`) |
| `CLOUDTPU_FUSE_GRAPHS` | `1` | `1` fuses contiguous supported GGML nodes + enables async pipelining |
| `CLOUDTPU_CACHE_CAP` | `4096` | LRU capacity of compiled `PJRT_LoadedExecutable*` and `SubgraphPlanCache` |
| `CLOUDTPU_MIN_ELEMENTS` | `0` | Minimum `MUL_MAT` element work threshold before offloading to TPU |
| `CLOUDTPU_VERBOSE` | `0` | `1` logs init, SPMD config, `plan_hits`/`plan_misses`, and PJRT cache stats; `3` dumps emitted StableHLO MLIR |
| `GGML_DISABLE_CLOUDTPU` | unset | Set `1` to disable `CLOUDTPU` device registration |
