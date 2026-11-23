# `ggml-cloudtpu` — Native C++17 `llama.cpp` Backend for Google Cloud TPUs

Clean-room Apache-2.0 GGML `GPU` backend (`CLOUDTPU`) targeting Google Cloud TPUs (`v5e`, `v6e`, `tpu7x`) directly over OpenXLA's C `PJRT_Api` (`dlopen(libtpu.so)` -> `GetPjrtApi()`).

**Zero Python, zero JAX, zero PyTorch, zero `vllm-tpu` at runtime (`MAPS_LIBPYTHON=0`).**

---

## Architecture (M1–M4)

| Component | File(s) | Responsibility |
| --- | --- | --- |
| **PJRT Runtime & Cache** | `pjrt_runtime.hpp`, `pjrt_runtime.cpp` | `dlopen(libtpu.so)` -> `GetPjrtApi()` (with opt-in `CLOUDTPU_ASYNC_COLLECTIVES=1` appending documented TPU async collective overlap flags to `LIBTPU_INIT_ARGS` prior to `libtpu.so` initialization), pure C++ `CompileOptionsProto` (optional typed `env_option_overrides` for `xla::DebugOptions` overrides without overwriting compiler defaults/`XLA_FLAGS`) + `ExecutableBuildOptionsProto` + `DeviceAssignmentProto` wire encoder, thread-safe LRU executable cache (`PJRT_LoadedExecutable*`) with eviction epoch, single-chip & multi-chip replica-SPMD execution (`execute_spmd` with monotonic `launch_id` and `non_donatable_input_indices`), HBM buffer H2D/D2H with explicit untiled C-order `host_layout`, `/proc/self/maps` `libpython` audit, and multi-device **Host Mock PJRT** (`CLOUDTPU_MOCK=1`, `CLOUDTPU_MOCK_DEVICES=1..16`) executing StableHLO IR + `dynamic_update_slice` + cross-chip collectives |
| **StableHLO Emitter** | `stablehlo_emitter.hpp`, `stablehlo_emitter.cpp` | Zero-dependency C++17 textual StableHLO MLIR emitter (`format="mlir"`) + deterministic structural cache key for TPU-native K-major N-minor quantized `MUL_MAT` (`[B3, B2, NB, 16/32, N]` + split-X zero-concat Q4 dot + BF16 MXU), `FLASH_ATTN_EXT` (GQA + mask + ALiBi + softcap), `RMS_NORM`, `ROPE` (Normal & NeoX + YaRN + `rope_freqs`), `SILU`, `SWIGLU`, `SOFT_MAX`, `GET_ROWS` (last-axis gather on quantized tables), `SET_ROWS` (single-token `dynamic_update_slice` + `{tf.aliasing_output}` KV donation, multi-row `scatter`), `ADD`, `MUL`, `SCALE`, `CPY`, `RESHAPE`, `TRANSPOSE`, `SLICE`, and StableHLO v1.0 collectives (`ALL_REDUCE_SUM`, `ALL_GATHER`, `REDUCE_SCATTER_SUM`) |
| **GGML Backend** | `ggml-cloudtpu.h`, `ggml-cloudtpu.cpp` | HBM-resident multi-chip `ggml_backend_buffer` slot manager (K-major N-minor `(8,128)` tile-aligned quantized HBM upload including native `S4` 0.5 B/weight packing & axis-aware SPMD sharding for `Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`, O(1) slot reuse fast-path, runtime-checked in-place multi-layer KV buffer donation, sub-slice view patching), O(N) `ggml_gallocr` liveness precomputation, zero-MLIR-re-emission `SubgraphPlanCache`, whole-model 1-graph fusion (`CLOUDTPU_MAX_FUSE_OPS=4096`, `CLOUDTPU_FUSE_ATTN=1` with in-slice KV SSA forwarding), `pad8` MXU sublane alignment (`CLOUDTPU_MM_PAD_M=8`), continuous-batching padded buckets (`CLOUDTPU_BATCH_BUCKETS=1,2,4,8,16,32,64,128,256,512` with KV-safe `-1` scatter padding), async pipelining (`ggml_backend_cloudtpu_synchronize`), and recursive `VIEW`/`PERMUTE`/`TRANSPOSE`/`RESHAPE` lowering |
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
5. **BF16 MXU contraction (`CLOUDTPU_MATMUL_BF16=1`, opt-in `CLOUDTPU_FLASH_ATTN_BF16=0`):** computes split-quantized affine `q * d + m` in F32 VPU precision and casts once to `BF16` for native MXU `MUL_MAT` contraction (`114/114` `Q4_K`, `102/102` `Q8_0`, `63/63` `Q6_K`, `102/102` `Q4_0`, `287/287` `ROPE`, `65/65` `SET_ROWS`, `56/56` `GET_ROWS` passed on live `v5e-1`), while keeping dense F16/F32 matmuls and `FLASH_ATTN_EXT` in F32 by default for strict GGML parity unless `CLOUDTPU_FLASH_ATTN_BF16=1` is opted in.
6. **Multi-chip async collective overlap (`CLOUDTPU_ASYNC_COLLECTIVES=1`, opt-in):** appends documented TPU async collective overlap flags (`--xla_tpu_enable_async_collective_fusion=true`, `--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true`, `--xla_tpu_enable_async_collective_fusion_multiple_steps=true`, `--xla_tpu_overlap_compute_collective_tc=true`, `--xla_enable_async_all_gather=true`) to `LIBTPU_INIT_ARGS` before `libtpu.so` loads (preserving any user-supplied flags without duplication, and leaving `CompileOptionsProto.env_option_overrides` free of non-`DebugOptions` keys) plus monotonic `launch_id` when `P > 1`.

| Model (GGUF) | Quant | Size | M3 `pp512` | M4B `pp512` | **M4C `pp512`** | M3 `tg64` | M4B `tg64` | **M4C `tg64`** (`FUSE_ATTN=0` / `1`) | M4C vs M4B `tg64` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-1.5B-Instruct` | `Q4_K_M` | `1.04 GiB` | `16,113.17` | `28,132.27` | **`22,220.93`** | `39.17` | `192.06` | `190.19` / **`209.88`** | **+9.3%** |
| `Qwen2.5-1.5B-Instruct` | `Q8_0` | `1.76 GiB` | `19,820.42` | `28,371.30` | **`29,662.17`** | `42.24` | `202.96` | `206.00` / **`210.63`** | **+3.8%** |
| `Meta-Llama-3.1-8B-Instruct` | `Q4_K_M` | `4.58 GiB` | `4,684.60` | `4,726.56` | **`6,524.26`** | `33.25` | `38.74` | `69.56` / **`70.24`** | **+81.3% (1.81x)** |
| `Meta-Llama-3.1-8B-Instruct` | `Q8_0` | `7.95 GiB` | `4,207.50` | `4,237.41` | **`6,963.31`** | `35.20` | `53.06` | **`60.33`** / `57.66` | **+13.7% (>62% peak HBM BW)** |

*Both `Qwen2.5-1.5B` and `Meta-Llama-3.1-8B` pass live `v5e-1` `llama-cli` coherence (`17 * 23 = 391`) with verified `/proc/<pid>/maps` `MAPS_LIBPYTHON=0` and `LDD_LIBPYTHON=0`.*

### 1-Graph Decode + Native `S4` + `pad8` + Continuous Batching (`TPU v5e-1`, `MAPS_LIBPYTHON=0`)

Four stacked optimizations close the single-stream and batched serving gap while keeping 100% pure StableHLO C++ portability (`LDD_LIBPYTHON=0`, `MAPS_LIBPYTHON=0`):
1. **Whole-model 1-graph decode (`CLOUDTPU_FUSE_ATTN=1`, `CLOUDTPU_MAX_FUSE_OPS=4096`):** In-slice KV SSA forwarding (`SET_ROWS` -> `FLASH_ATTN_EXT` views) collapses `Llama-3.1-8B` from `65` PJRT launches/token to **`1` subgraph/eval** (`evals=65, subgraphs=65`) with full `2 * n_layers` KV buffer donation (`kv_donation_skipped=0`).
2. **Native StableHLO `S4` single-read Q4 matmul (`CLOUDTPU_Q4_MODE=i4`):** Packs `Q4_0` (`q-8`) and `Q4_K` (`q-8`, preserving exact unshifted `m`) directly into 0.5 B/weight `PJRT_Buffer_Type_S4` with factored `((dot_b + 8*xsum) * d) + (xsum * m)` lowering (`M <= 32`) and in-loop dequant (`M > 32`), plus opt-in native `BF16` `Q4_K`/`Q6_K` scales (`CLOUDTPU_Q4K_BF16_SCALES=1`, default `0` to preserve full F32 `GET_ROWS` scale precision) eliminating `F16->F32->BF16` XLA roundtrips and `125 MiB` `lm_head` HBM temps (`Q4_K/Q6_K/Q4_0/Q8_0` `MUL_MAT` `57/57`, full `MUL_MAT` `1,361/1,361`, `SET_ROWS` `65/65`, `GET_ROWS` `56/56`, `ROPE` `287/287`, `RMS_NORM` `61/61`, `SOFT_MAX` `107/107`, `ADD` `153/153`, `MUL` `140/140`, `SCALE` `12/12`, `CPY` `157/157`, `CONT` `89/89` on `v5e-1`).
3. **MXU `(8,128)` `pad8` small-M alignment (`CLOUDTPU_MM_PAD_M=8`):** Zero-pads `M < 8 -> 8` before `Q4_*_I4` and `Q6_K` `dot_general` so single-token decode routes to the TPU MXU instead of VPU (`+16.4%` `1.5B`, `+20.6%` `8B` over unpadded 1-graph).
4. **Continuous-batching padded buckets (`CLOUDTPU_BATCH_BUCKETS=1,2,4,8,16,32,64,128,256,512`):** Dataflow token-axis analysis pads dynamic batches to static buckets (with `-1` KV scatter sentinel indices preventing any padded write corruption) — verified across odd `llama-bench -p 3,5,6,7,9,10,11` (`3` cache entries total for `7` shapes, `bucketed=2` each) and concurrent `llama-server -np 16 -cb -kvu`.

| Model (GGUF) | Quant | Baseline M4C `pp512` | **New `pp512`** | Baseline M4C `tg64` | **New 1-Graph `tg64`** | Gain vs M4C `tg64` | `subgraphs/eval` | `kv_donation_skipped` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-1.5B` | `Q4_K_M` | `21,836.23` | **`30,176.38`** | `210.47` | **`244.97`** | **+16.4%** | `1.0` | `0` |
| `Llama-3.1-8B` | `Q4_K_M` | `6,555.99` | **`7,021.04`** | `71.03` | **`83.25`** (`84.15` @ 3-sub) | **+17.2%** | `1.0` | `0` |

#### Live Continuous Batching (`llama-server -np 16 -cb -kvu` w/ `LLAMA_KV_ROWS_DISABLE=1`, `TPU v5e-1`, `MAPS_LIBPYTHON=0`)

| Model (`Q4_K_M`) | Concurrency (`c`) | Steady Wall `tok/s` | Steady Decode `tok/s` | Coherence & KV Check |
| --- | ---: | ---: | ---: | --- |
| `Llama-3.1-8B` | `1` | `70.54` | `72.35` | Coherent, `0` skips |
| `Llama-3.1-8B` | `4` | `242.75` | `256.31` | Coherent all 4 (`3.54x`) |
| `Llama-3.1-8B` | `8` | `66.25` (1 prefill compile) | `473.52` | Coherent all 8 (`6.54x` decode) |
| `Llama-3.1-8B` | **`16`** | **`756.55`** (`1.354s`) | **`787.79`** | Coherent all 16 (**`10.9x` decode**), `bucketed=83`, `kv_donation_skipped=0` |
| `Qwen2.5-1.5B` | `1` / **`16`** | `196.32` / `148.97` (1 prefill compile) | `203.76` / **`1,331.86`** | Coherent all 16, `bucketed=14`, `kv_donation_skipped=0` |

*Note: Optional free-tier Colab `V6E1` single-chip probe returned service unavailable (`UNAVAILABLE`). Live paid `v6e-8` hardware validation results and compatibility fixes are recorded below.*

#### Live `v6e-8` Verification & Compatibility Notes (`TPU v6 lite` `Aug 15 2025` `libtpu.so`, `MAPS_LIBPYTHON=0`, `LDD_LIBPYTHON=0`)
- **Runtime & Op Parity (`CLOUDTPU_MOCK=0`):**
  - `test_cloudtpu_host` (`CLOUDTPU_MOCK=0`): **`ALL PASSED (0 failures)`** on live `v6e-8` — single-chip live `libtpu.so` hardware tests (`S4`/`U4` H2D/D2H roundtrip after normalizing zero-extended `S4` `PJRT_Buffer_ToHostBuffer` nibbles via `sign_extend_nibble` in `pjrt_runtime.cpp`, `Q4_0_I4`/`Q4_K_I4` single-read `S4`/`U4`/`bitcast` `MUL_MAT`, `GET_ROWS`+`MUL_MAT` coexistence, `BF16` scales + `pad8`, and `P=1` collectives) passed on live `TPU v6 lite`, while the multi-chip Q4 sharding, multi-layer 1-graph KV donation, and padded continuous-batching unit tests inside `test_cloudtpu_host` explicitly select the in-process Host Mock (live hardware verification of multi-chip SPMD, 1-graph decode, and continuous batching is recorded via `test-backend-ops`, `llama-bench`, and `llama-server` below).
  - `test-backend-ops -b CLOUDTPU` (`CLOUDTPU_SPMD_CHIPS=8`): `SET_ROWS` and `RMS_NORM` passed; quantized `MUL_MAT` (`q4_K`/`q6_K`/`q4_0`/`q8_0`) passed **`381/381`**; `GET_ROWS` (`CLOUDTPU_SPMD_CHIPS=1`, default `CLOUDTPU_Q4K_BF16_SCALES=0`, `q4_K`/`q6_K`/`q4_0`) passed **`15/15`** (opt-in `CLOUDTPU_Q4K_BF16_SCALES=1` trades `BF16` scale rounding `~4e-6` vs `test-backend-ops`'s `1e-7` lookup tolerance).
- **Single-Chip `v6e-1` (`CLOUDTPU_SPMD_CHIPS=1 -sm none -mg 0`) & 8-Chip `v6e-8` (`CLOUDTPU_SPMD_CHIPS=8 CLOUDTPU_ASYNC_COLLECTIVES=0`):**
  - Pinning `-sm none -mg 0` keeps single-chip runs on chip 0 instead of `llama.cpp` layer-splitting across all 8 exposed devices: `Qwen2.5-1.5B Q4_K_M` achieved **`pp512 = 16,262.01` / `tg64 = 73.92`** (`+35.6%` `tg64` vs M4A `54.52`); `Llama-3.1-8B Q4_K_M` achieved **`pp512 = 7,557.94` / `tg64 = 52.98`** and coherent `16/16` continuous batching via `llama-server -np 16 -cb -kvu` (`50.00` decode at `c=1`, `20.30` decode at `c=16`, `MAPS_LIBPYTHON=0`).
  - Keeping **`CLOUDTPU_ASYNC_COLLECTIVES=0`** by default in `ggml-cloudtpu.cpp` avoids the legacy `Aug 15 2025` `libtpu.so` `PJRT_Client_Compile: MegaChip currently works correctly only when base is zero` failure on 8-chip 1-graph execution (`Qwen2.5-1.5B` `-p 4 -n 4`: `pp4 = 61.25` / `tg4 = 28.79`, `2.54x` `tg4` vs `CLOUDTPU_FUSE_ATTN=0`).
  - **Host `libtpu` Version Tip:** Ubuntu 22.04 Cloud TPU VMs ship Python 3.10, causing `pip install libtpu` to pull a legacy 2025 `libtpu.so` (`libtpu >= 0.0.50` wheels require Python `>=3.11`). Extracting `libtpu/libtpu.so` directly from the PyPI `cp312` `manylinux_2_31_x86_64` wheel and setting `LIBTPU_PATH` avoids any Python dependency at runtime (`MAPS_LIBPYTHON=0`).

#### Live `v6e-8` Verification (`libtpu 0.0.53.1`, `Nov 17 2026` `TFRT TPU v6 lite`, `fedad85`, `MAPS_LIBPYTHON=0`, `LDD_LIBPYTHON=0`)
- **Clean `cp312` `libtpu 0.0.53.1` Extraction (Zero Runtime Python):**
  ```bash
  python3 -c '
  import io, json, urllib.request, zipfile
  meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/libtpu/0.0.53.1/json", timeout=30))
  whl = [u["url"] for u in meta["urls"] if "cp312" in u["filename"] and "x86_64" in u["filename"]][0]
  zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(whl, timeout=120).read()))
  so = [n for n in zf.namelist() if n.endswith("/libtpu.so")][0]
  open("libtpu-0.0.53.1.so", "wb").write(zf.read(so))
  '
  export LIBTPU_PATH="$PWD/libtpu-0.0.53.1.so"
  ```
- **Standalone Suite & `ASYNC_COLLECTIVES=1` Findings (`CLOUDTPU_MOCK=0`):**
  - `test_cloudtpu_host` (`CLOUDTPU_MOCK=0`, `LIBTPU_PATH=libtpu-0.0.53.1.so`): `28` passing test groups with `1` assertion failure at `tests/test_cloudtpu_host.cpp:1059` (`Opt-in BF16 FLASH_ATTN_EXT should reflect BF16 rounding (0.5f)`), `MAPS_LIBPYTHON=0`.
  - **`CLOUDTPU_ASYNC_COLLECTIVES=1` on `libtpu 0.0.53.1` `v6e` (`ghostlite`):** `PJRT_Client_Create` fails initialization with `Continuation fusion for AllGather is enabled on platform ghostlite (variant: ), but it is not supported/recommended on platforms other than TPU v5e... Please disable it with xla_tpu_enable_async_collective_fusion=false` (`0` devices exposed, causing `llama-bench` to fall back to host CPU). Keep **`CLOUDTPU_ASYNC_COLLECTIVES=0`** (default) on `v6e`.
- **End-to-End `v6e-8` Benchmarks (`llama-bench -ngl 99 -fa 1 -p 512 -n 64 -r 3` & `llama-cli`, `libtpu 0.0.53.1`):**
  *(3-iteration means include run-1 XLA cold JIT compile in stddev.)*

| Model (GGUF `Q4_K_M`) | Size | Chips (`P`) | `pp512` (tok/s) | `tg64` (tok/s) | Speedup vs Legacy `v6e` `libtpu.so` | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `Qwen2.5-1.5B-Instruct` | `934.69 MiB` | `1` (`-sm none -mg 0`) | `16,651.60 ± 3,111.77` | **`183.01 ± 60.32`** | **2.48x `tg64`** (`73.92`) | `llama-cli` coherent (`17 * 23 = 391`) |
| `Qwen2.5-1.5B-Instruct` | `934.69 MiB` | `8` (`SPMD`) | `11,424.67 ± 787.76` | **`68.86 ± 20.19`** | **5.68x `tg64`** (`12.13` M4A) | `llama-cli` coherent (`391`) |
| `Meta-Llama-3.1-8B-Instruct` | `4.58 GiB` | `1` (`-sm none -mg 0`) | `7,594.14 ± 1,681.51` | **`91.62 ± 28.00`** | **1.73x `tg64`** (`52.98`) | `llama-cli` coherent (`17 * 23 = 391`) |
| `Meta-Llama-3.1-8B-Instruct` | `4.58 GiB` | `8` (`SPMD`) | `7,974.23 ± 2,478.81` | **`65.04 ± 8.00`** | — | `llama-cli` coherent (`17 * 23 = 391`) |
| `Qwen3-32B` | `18.40 GiB` | `8` (`SPMD`) | **`3,609.05 ± 171.84`** | **`29.00 ± 1.54`** | **4.03x `tg64`** (`7.20` M4A) | `llama-cli` coherent (`17 * 23 = 391`) |

  - `Llama-3.1-8B P=1` continuous batching (`llama-server -ngl 99 -fa 1 -sm none -mg 0 -np 16 -cb -kvu -c 8192` w/ `LLAMA_KV_ROWS_DISABLE=1`, `MAPS_LIBPYTHON=0`): `16/16` coherent replies (`sum_decode_tok_s` `8.51` / `17.24` / `22.26` / `28.19` and cold-compile wall throughput `4.15` / `11.43` / `2.50` / `3.71` tok/s across `c=1, 4, 8, 16`; `c=1` timing check: `8.51` decode `tok/s`).
  - `Llama-3.3-70B-Instruct` (`Q4_K_M`, `39.59 GiB`, `P=8`): GGUF downloaded and entered weight load (`Loading model...`), deferred before `llama-bench` completion due to the VM time cap.

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

| Target Topology | Logical Chiplets (`P`) | Total HBM | `Qwen3-32B` (`Q4_K_M` / `BF16`) per chiplet | `Llama-3.3-70B` (`Q4_K_M` / `Q8_0` / `BF16`) per chiplet | Verification Status |
| --- | ---: | ---: | --- | --- | --- |
| **`v6e-8`** (Trillium) | `8` (`8` chips) | `256 GiB` (`32 GiB/chip`) | `~2.5 GiB` / `~8.1 GiB` | `~5.3 GiB` / `~9.4 GiB` / `~17.6 GiB` | **Verified live on `v6e-8` silicon** (`test_cloudtpu_host`, `3,820/3,820` `test-backend-ops`, `Qwen3-32B`, `Llama-3.3-70B` `39.59 GiB`, `MAPS_LIBPYTHON=0`) |
| **`tpu7x-standard-1t`** (Ironwood) | `2` (`1` chip package, D2D pair) | `192 GiB` (`96 GiB/chiplet`) | `~10.0 GiB` / `~32.5 GiB` | `~21.2 GiB` / `~37.5 GiB` / `~70.5 GiB` | Verified on `tpu7x-1t` dual-chiplet Host Mock (`f8E4M3FN` + D2D pairing); live Day-1 runbook in [`docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md`](../docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md) |
| **`tpu7x-standard-4t`** (Ironwood) | `8` (`4` chips × `2` chiplets) | `768 GiB` (`96 GiB/chiplet`, `192 GiB/chip`) | `~2.5 GiB` / `~8.1 GiB` | `~5.3 GiB` / `~9.4 GiB` / `~17.6 GiB` | Verified on `tpu7x-4t` 8-chiplet D2D-paired Host Mock (`f8E4M3FN` + Hamiltonian pairing); live Ironwood Day-1 proof in [`docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md`](../docs/cloud-tpu/ironwood-tpu7x-day1-runbook.md) |

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

# Continuous-batching serving on TPU with unified KV cache + static padded buckets:
LLAMA_KV_ROWS_DISABLE=1 ./llama-cpp-cloudtpu/build-host/bin/llama-server \
  -m model-q4_k_m.gguf -ngl 99 -fa 1 -np 16 -cb -kvu -c 8192 --port 8089

# Pinning a single chip on a multi-chip TPU host (prevents ggml layer-splitting across CLOUDTPU0..N):
CLOUDTPU_SPMD_CHIPS=1 ./llama-cpp-cloudtpu/build-host/bin/llama-bench \
  -m model-q4_k_m.gguf -ngl 99 -fa 1 -sm none -mg 0 -p 512 -n 64
```

---

## Runtime Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLOUDTPU_MOCK` | `0` on Linux, `1` on macOS/Windows | `1` runs in-process Host Mock PJRT executing emitted StableHLO IR |
| `CLOUDTPU_MOCK_DEVICES` | `1` | Virtual logical TPU chiplet count in Host Mock (`2` = `tpu7x-1t`, `8` = `v6e-8` or `tpu7x-4t` with `CLOUDTPU_MOCK_PROFILE`) |
| `CLOUDTPU_MOCK_PROFILE` | `auto` | Host Mock hardware topology profile (`auto`, `v5e-1`, `v6e-8`, `tpu7x-1t`, `tpu7x-4t`) |
| `CLOUDTPU_SPMD_CHIPS` | `1` (per-chip devices) | Opt-in replica-SPMD Tensor Parallelism: `1` exposes per-chip devices (`CLOUDTPU`, `CLOUDTPU1`, ...); `auto` uses all discovered chips; `2..16` sets an explicit mesh size |
| `CLOUDTPU_SPMD_MIN_ROWS` | `256` | Minimum 2D weight matrix rows/cols required to shard across SPMD chips |
| `CLOUDTPU_MAX_FUSE_OPS` | `4096` | Maximum GGML ops fused into a single StableHLO subgraph executable (`1..16384`) |
| `CLOUDTPU_FUSE_ATTN` | `1` | `1` fuses `FLASH_ATTN_EXT` + KV updates into 1 PJRT graph/token (`0` splits at attention) |
| `CLOUDTPU_DONATE_KV` | `1` | `1` enables in-place KV cache HBM donation via `dynamic_update_slice` + `{tf.aliasing_output}` |
| `CLOUDTPU_MATMUL_BF16` | `1` | `1` casts dequantized split-quantized weights (`Q8_0`/`Q4_0`/`Q4_K`/`Q6_K`) & activations to `BF16` for TPU MXU `MUL_MAT` contraction (dense F16/F32 matmuls stay F32) |
| `CLOUDTPU_MATMUL_FP8` | `0` | Quantized `MUL_MAT` `f8E4M3FN` MXU lowering (`0` = off / BF16 fallback, `auto` = enabled automatically on Ironwood `tpu7x`, `1` = force `f8E4M3FN` with dynamic absmax scaling) |
| `CLOUDTPU_EMIT_SDY_MESH` | `0` | Opt-in (`1`) emits D2D-aware Shardy `sdy.mesh` module declaration aligned to Hamiltonian intra-package pairing |
| `CLOUDTPU_FLASH_ATTN_BF16` | `0` | Opt-in (`1`) casts `FLASH_ATTN_EXT` Q/K/probs/V matmuls to `BF16` (`0` keeps F32 attention for strict GGML parity) |
| `CLOUDTPU_Q4_MODE` | `i4` | Q4 storage/kernel path: `i4` (`S4` 0.5 B/w factored dot), `u4`, `bitcast` (`U8->S4`), `split`, `concat` |
| `CLOUDTPU_Q4_SPLIT_DOT` | `1` | Legacy toggle (`1` -> `split`, `0` -> `concat`) when not using `CLOUDTPU_Q4_MODE` |
| `CLOUDTPU_Q4K_BF16_SCALES` | `0` | Opt-in (`1`) stores `Q4_K`/`Q6_K` effective scales in hardware `BF16` (`0` = default F32 scales preserving strict `GET_ROWS` parity; BF16 keeps full F32 exponent range) |
| `CLOUDTPU_MM_PAD_M` | `8` | Pads small `M < N` `Q4_*_I4` and `Q6_K` matmuls to TPU MXU `(8,128)` sublane width (`0`/`1` disables) |
| `CLOUDTPU_BATCH_BUCKETS` | `1,2,4,8,16,32,64,128,256,512` | Static continuous-batching token buckets (`0`/empty disables padding) |
| `CLOUDTPU_ASYNC_COLLECTIVES` | `0` | Opt-in (`1`) appends documented TPU async collective overlap flags to `LIBTPU_INIT_ARGS` prior to `libtpu.so` init (off by default for legacy `v6e` `libtpu.so` MegaChip compatibility) |
| `LIBTPU_PATH` / `TPU_LIBRARY_PATH` | Auto-resolved `libtpu.so` | Explicit path to `libtpu.so` (or future clean-room `libopentpu.so`) |
| `CLOUDTPU_FUSE_GRAPHS` | `1` | `1` fuses contiguous supported GGML nodes + enables async pipelining |
| `CLOUDTPU_CACHE_CAP` | `4096` | LRU capacity of compiled `PJRT_LoadedExecutable*` and `SubgraphPlanCache` |
| `CLOUDTPU_MIN_ELEMENTS` | `0` | Minimum `MUL_MAT` element work threshold before offloading to TPU |
| `CLOUDTPU_VERBOSE` | `0` | `1` logs init, SPMD config, `plan_hits`/`plan_misses`, and PJRT cache stats; `3` dumps emitted StableHLO MLIR |
| `GGML_DISABLE_CLOUDTPU` | unset | Set `1` to disable `CLOUDTPU` device registration |
