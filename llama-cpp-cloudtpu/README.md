# `ggml-cloudtpu` — Native C++17 `llama.cpp` Backend for Google Cloud TPUs

Clean-room Apache-2.0 GGML `GPU` backend (`CLOUDTPU`) targeting Google Cloud TPUs (`v5e`, `v6e`, `tpu7x`) directly over OpenXLA's C `PJRT_Api` (`dlopen(libtpu.so)` -> `GetPjrtApi()`).

**Zero Python, zero JAX, zero PyTorch, zero `vllm-tpu` at runtime (`MAPS_LIBPYTHON=0`).**

---

## Architecture (M1–M4)

| Component | File(s) | Responsibility |
| --- | --- | --- |
| **PJRT Runtime & Cache** | `pjrt_runtime.hpp`, `pjrt_runtime.cpp` | `dlopen(libtpu.so)` -> `GetPjrtApi()`, pure C++ `CompileOptionsProto` + `DeviceAssignmentProto` wire encoder, thread-safe LRU executable cache (`PJRT_LoadedExecutable*`) with eviction epoch, single-chip & multi-chip replica-SPMD execution (`execute_spmd`), HBM buffer H2D/D2H with explicit untiled C-order `host_layout`, `/proc/self/maps` `libpython` audit, and multi-device **Host Mock PJRT** (`CLOUDTPU_MOCK=1`, `CLOUDTPU_MOCK_DEVICES=1..16`) executing StableHLO IR + cross-chip collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `replica_id`, `dynamic_slice`) |
| **StableHLO Emitter** | `stablehlo_emitter.hpp`, `stablehlo_emitter.cpp` | Zero-dependency C++17 textual StableHLO MLIR emitter (`format="mlir"`) + deterministic structural cache key for `MUL_MAT` (`F32`/`F16`/`BF16`/`Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`, replicated, `SHARD_N`, and `SHARD_K`), `FLASH_ATTN_EXT` (GQA + mask + ALiBi + softcap), `RMS_NORM`, `ROPE` (Normal & NeoX + YaRN + `rope_freqs`), `SILU`, `SWIGLU`, `SOFT_MAX`, `GET_ROWS`, `SET_ROWS` (`stablehlo.scatter`), `ADD`, `MUL`, `SCALE`, `CPY`, `RESHAPE`, `TRANSPOSE`, `SLICE`, and StableHLO v1.0 collectives (`ALL_REDUCE_SUM`, `ALL_GATHER`, `REDUCE_SCATTER_SUM`) |
| **GGML Backend** | `ggml-cloudtpu.h`, `ggml-cloudtpu.cpp` | HBM-resident multi-chip `ggml_backend_buffer` slot manager (lossless split & sharded HBM upload for `Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`/`F16`/`BF16`/`F32`, O(1) slot reuse fast-path, sub-slice view patching), O(N) `ggml_gallocr` liveness precomputation, zero-MLIR-re-emission `SubgraphPlanCache`, extended fusion (`CLOUDTPU_MAX_FUSE_OPS=64`) with automatic Megatron-LM 1-collective FFN detection (`Gate`/`Up` `SHARD_N` -> local `SwiGLU` -> `Down` `SHARD_K` + single `all_reduce`), async pipelining (`ggml_backend_cloudtpu_synchronize`), and recursive `VIEW`/`PERMUTE`/`TRANSPOSE`/`RESHAPE` lowering |
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

### M3 -> M4B End-to-End GGUF Throughput (`TPU v5e-1`, `llama-bench -ngl 99 -fa 1`)

Eliminating host MLIR re-emission (`SubgraphPlanCache`), replacing per-subgraph $O(N^2)$ overlap scans with single-pass $O(N)$ cgraph liveness, raising `CLOUDTPU_MAX_FUSE_OPS` (`16 -> 64`) with safe `SET_ROWS` fusion, and async pipelining boosted `Qwen2.5-1.5B` decode by **4.8x–4.9x** and lifted `Llama-3.1-8B Q8_0` to **>51% of `v5e-1` peak HBM bandwidth (`422 GB/s`)**:

| Model (GGUF) | Quant | Size | M3 `pp128` | **M4B `pp128`** | M3 `pp512` | **M4B `pp512`** | M3 `tg64` | **M4B `tg64`** | `tg64` Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `4,944.70` | **`15,541.37`** | `16,113.17` | **`28,132.27`** | `39.17` | **`192.06`** | **4.90x** |
| `Qwen2.5-1.5B-Instruct` | `Q8_0` | `1.76 GiB` | `5,445.35` | **`16,511.84`** | `19,820.42` | **`28,371.30`** | `42.24` | **`202.96`** | **4.80x** |
| `Meta-Llama-3.1-8B-Instruct` | `Q4_K_M` | `4.58 GiB` | `1,957.64` | **`1,977.52`** | `4,684.60` | **`4,726.56`** | `33.25` | **`38.74`** | **1.17x** |
| `Meta-Llama-3.1-8B-Instruct` | `Q8_0` | `7.95 GiB` | `1,910.83` | **`1,938.66`** | `4,207.50` | **`4,237.41`** | `35.20` | **`53.06`** | **1.51x** |

*Note: `Meta-Llama-3.1-8B` carries `rope_freqs.weight` on CPU, causing `ggml_backend_sched` to emit 65 splits per token vs 1 split on `Qwen2.5-1.5B`; both models generate coherent verified `llama-cli` (`-c 4096`, `17 * 23 = 391`) and `llama-server` responses with `MAPS_LIBPYTHON=0`. Full `FLASH_ATTN_EXT` (`1697/1697`), `SET_ROWS` (`65/65`), `RMS_NORM` (`61/61`), and `MUL_MAT` (`44/44`) parity verified on live `TPU v5 lite`.*

### M4A Multi-Chip SPMD Live Silicon Proof (`v6e-8` Trillium, `MAPS_LIBPYTHON=0`)

When opted in via `CLOUDTPU_SPMD_CHIPS=auto` (or an explicit count such as `CLOUDTPU_SPMD_CHIPS=8`) with `P > 1` TPU chips, `ggml-cloudtpu` exposes a single unified `CLOUDTPU` device reporting aggregated mesh HBM (`P * min_chip_hbm`), compiles explicit replica-SPMD StableHLO v1.0 collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `replica_id` + `dynamic_slice`) with `DeviceAssignmentProto` (`num_replicas=P, num_partitions=1`), and automatically shards 2D weights by `1/P` across chips (`SHARD_K` for `ffn_down`/`token_embd`, `SHARD_N` for `Q`/`K`/`V`/`O`/`ffn_gate`/`ffn_up`/`output`, fusing Megatron-LM FFN blocks down to a single `all_reduce`). Leaving `CLOUDTPU_SPMD_CHIPS` unset (default `1`) preserves per-chip devices (`CLOUDTPU`, `CLOUDTPU1`, ...).

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
| `CLOUDTPU_MAX_FUSE_OPS` | `64` | Maximum GGML ops fused into a single StableHLO subgraph executable |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | Auto-resolved `libtpu.so` | Explicit path to `libtpu.so` (or future clean-room `libopentpu.so`) |
| `CLOUDTPU_FUSE_GRAPHS` | `1` | `1` fuses contiguous supported GGML nodes + enables async pipelining |
| `CLOUDTPU_CACHE_CAP` | `4096` | LRU capacity of compiled `PJRT_LoadedExecutable*` and `SubgraphPlanCache` |
| `CLOUDTPU_MIN_ELEMENTS` | `0` | Minimum `MUL_MAT` element work threshold before offloading to TPU |
| `CLOUDTPU_VERBOSE` | `0` | `1` logs init, SPMD config, `plan_hits`/`plan_misses`, and PJRT cache stats; `3` dumps emitted StableHLO MLIR |
| `GGML_DISABLE_CLOUDTPU` | unset | Set `1` to disable `CLOUDTPU` device registration |
