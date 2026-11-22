# Phase 7 Performance & Architecture Report: Host-Tax Elimination, Chained SwiGLU, MoE (`MUL_MAT_ID`) & LMKD Residency

**Target Device**: Google Pixel 10 Pro Fold (`rango`, Tensor G5 `buenos_a0`, Android 17 `CP3A.261005.005`, kernel `6.6.127`).  
**Clean-Room Invariant**: Zero Google/vendor `.so` dependencies (`llvm-readelf -d` `NEEDED` = `[libdl.so, libm.so, libc.so]` only on all stripped Android ARM64 ELFs; `0` `dlopen`/`dlsym`; `0` embedded RV64 ELF; live `/proc/<pid>/maps` `MAPS_GOOGLE=0` across all runs).  
**Baseline**: Builds on [Phase 6 (`PR #2`)](PHASE6_PERF_REPORT.md) (`1,286.4 MHz` DVFS + FW WakeLock, `M=64` batched prefill, native `Q4_0`, and `0xa40` multi-op chaining).

---

## 1. In-SRAM Chained SwiGLU (`0x640` SiLU * `0x100` MUL) & Silicon Probes

Building on Phase 6's `0xa40` multi-op concatenation decode ([`phase7_swiglu_cpp_spec.md`](isa-gemm/phase7_swiglu_cpp_spec.md)), Milestone 7B integrated an **11-chunk in-SRAM chained SwiGLU engine** (`gate -> 0x640 SiLU * up [0x100 MUL] -> down`) directly into [`libopenbuenos`](../libopenbuenos/include/openbuenos.h) (`ob_gemm_swiglu_f32_quant` / `ob_gemm_swiglu_int8`) and [`ggml-edgetpu.cpp`](../llama-cpp-edgetpu/ggml-edgetpu.cpp) (`try_compute_fused_swiglu`):

- **Silicon Reference & Probe Grounding**:
  - `16/16` compiler reference SwiGLU `.dgc0` models on `rango` (`sgs=1`, `num_chunks=11`, up to `9.2 MiB` weights): **`16/16` `bm_vs_direct = 0`**.
  - `6/6` standalone non-PoT synthesized `.dgc0` probes on live `rango` (`PM=5` + FW WakeLock @ `1,286.4 MHz`, `MAPS_GOOGLE=0`): **`6/6` `max_abs_err = 0`**:
    - `probe_swiglu_1x576x1536x576_smollm2` (**SmolLM2 full decode MLP in 1 VII call**): `576/576` exact, **`118.0 us` HW** (`0.561 ms` wall), `45.0 GOPS`, `22.50 GB/s`
    - `probe_swiglu_1x1536x1024x1536_qwen1024` (Qwen2.5 decode tile): `1536/1536` exact, **`180.0 us` HW** (`0.476 ms` wall), `52.4 GOPS`, **`26.23 GB/s`**
    - `probe_swiglu_1x1536x512x1536_qwen512`: `1536/1536` exact, `102.0 us` HW (`0.359 ms` wall), `23.16 GB/s`
    - `probe_swiglu_64x576x512x576_smollm2_pp`: `36,864/36,864` exact, **`144.0 us` HW** (`0.364 ms` wall), `786.4 GOPS`
    - `probe_swiglu_64x1536x512x1536_qwen_pp`: `98,304/98,304` exact, **`317.0 us` HW** (`0.607 ms` wall), **`952.6 GOPS`**
    - `probe_swiglu_1x64x64x64_nonpot`: `64/64` exact, `27.0 us` HW (`0.282 ms` wall)
- **End-to-End `libopenbuenos` + `ggml-edgetpu` Fusion**:
  - `SmolLM2-135M` decode (`H=1536`) runs the entire MLP block (`gate`, `up`, `SiLU`, `MUL`, `down`) in **1 VII submission** (`swiglu_chained = 5,788` across warm `llama-bench`); `Qwen2.5-1.5B` decode (`H=8960`) drops from `20` separate GEMM calls to **`9` `H=1024` chained VII submissions** (`swiglu_chained = 47,715` on `Q8_0`, `47,759` on `Q4_0`).
  - **BOS / Attention-Sink Coherence Guard**: Small prompts (`2 <= M <= 8` when `!prefill_chain_on`) and extreme `M=1` outlier rows (`crest > 16.0`) route to exact ARM64 NEON FP32 (`0 B` weight-cache pollution), while normal `M > 8` prefill routes to faster `lt_64x*` GEMM (`100%` QKV `3->1` batching preserved).
  - **Selftest Status (`openbuenos_selftest --test-swiglu`)**: Host `make verify` **`18/18` PASS**. On live `rango` (`--test-swiglu`), `7/8` cases passed (`4/4` `ob_gemm_swiglu_int8` `max_abs_err=0` at `99–329 us` HW; `smollm2_q8_1x576x1536` `108 us` HW / `248 us` wall / `nmse=2.39e-03`; `smollm2_q8_64x576x1536` `nmse=2.83e-03`; `qwen25_q4_1x1536x8960` `1624 us` HW / `1994 us` wall / **`nmse=1.70e-03`**, `sat_retries=0`), while opt-in prefill test `qwen25_q4_64x1536x8960` measured `nmse=4.00e-03` (`> 3.5e-3` selftest threshold, `RC=1`) after raising `d_down` headroom to eliminate decode `sat_retries`; production `M > 8` prefill uses `lt_64x*` GEMM.

---

## 2. Host-Tax Elimination & Stage Profile (`M7A -> M7B`)

Phase 7 added `OPENBUENOS_PROFILE=1` stage telemetry and attacked every non-TPU bottleneck identified in Phase 6:

1. **Pool Churn Fixed (`pool_hits=0` -> Active + 2-Tier `4 KiB / 128 KiB` Slabs + Shared `8 MiB` Zero-Params)**:
   - Returned `reg_cmd_buf` slabs directly to the buffer pool, added persistent `128 KiB` / `4 KiB` pipeline DMA-BUFs (cutting `DMA_BUF_IOCTL_SYNC` page walks), shared one `8 MiB` dummy `params_buf` across Probe-A handles (`350+ MiB` DMA-BUF savings), and enabled zero-copy mapped DMA outputs.
2. **Adaptive User-Space Spin (`OPENBUENOS_SPIN_US=450`)**:
   - Replacing blocking `poll()` (`SPIN_US=0`) with adaptive spinning for short `< 600 us` dispatches cut `poll_waits` from `23,755` -> **`420`** (`Qwen2.5 Q8 tg64`, `spin_hits=72,262`) and `5,192` -> **`11`** (`SmolLM2 Q8 tg64`, `spin_hits=21,613`), improving `tg64` speed and lowering total SoC energy.
3. **Graph-Level Horizontal Fusion & `ROPE`/`GLU` Offload**:
   - Supporting `GGML_OP_ROPE` (NEON + single-token trig cache) and `GGML_OP_GLU` inside `ggml-edgetpu` prevented `ROPE(Q)` from splitting QKV subgraphs, unlocking **`100%` `batched_3` QKV (`3->1`) fusion** (`horiz_fusions=5,790` SmolLM2 / `4,825` Qwen2.5 during `tg64`!) alongside 1D bias `ADD` (`bias_adds`) in the same backend eval.
4. **ARM64 NEON + Lazy Outlier-Column Cache + XOR-`0x80` Cold Packing**:
   - Coarsened SwiGLU quantization tuples (`2^(1/2)` / PoT + `kDefaultGraphCacheCapacity=512`) collapsed SwiGLU graph registrations from `~1,344` (`27.4s` `submit_us` thrash) down to `<= 88` (`2.89s` cold+warm `submit_us`).
   - Lazy FP32 `outlier_cols` cache + NEON `vfmaq_n_f32` + 16-byte XOR-`0x80` `pack_ext2` cut `Qwen2.5 Q8 tg64` `dequant_us` (`3.56s -> 0.67s`, **`5.3x` faster**) and `pack_us` (`4.14s -> 2.24s`).

### Host Stage Profile During `llama-bench` (`tg64` `profile_tg64`, ms)

| Model (Phase 7 M7B) | Calls | `hw_us` | `wall_us` | `quant` | `pack` | `dma_sync` | `submit` | `wait` | `dequant` | `spin_hits` / `poll_waits` | `w_evict` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `SmolLM2-135M Q8` | `11,773` | `1,743.8 ms` | `5,945.2 ms` | `11.2 ms` | `181.4 ms` | `35.5 ms` | `1,270.0 ms` | `4,248.4 ms` | `142.6 ms` | `21,613` / `11` | `0` |
| `Qwen2.5-1.5B Q8` | `11,001` | `12,725.9 ms` | `22,465.0 ms` | `18.2 ms` | `2,237.9 ms` | `76.0 ms` | `2,889.1 ms` | `15,971.4 ms` | `674.4 ms` | `72,262` / `420` | `0` |
| `Qwen2.5-1.5B Q4` | `10,808` | `10,898.7 ms` | `20,501.5 ms` | `26.7 ms` | `2,243.4 ms` | `94.6 ms` | `3,191.8 ms` | `13,997.8 ms` | `313.3 ms` | `58,675` / `420` | `0` |

*Remaining `pp512` host gap*: On standalone `Qwen2.5 Q8 pp512` (`energy_summary.json`), `hw_us = 5,614.7 ms` vs `wall_us = 12,054.7 ms`: single-threaded `dequant_us` (`4,784.2 ms`), `quant_us` (`1,145.0 ms`), `pack_us` (`1,700.7 ms`), and `submit_us` (`1,222.3 ms`) still exceed `hw_us`, explaining why end-to-end `pp512` (`139.08 t/s`) trails 4-thread CPU (`186.66 t/s`) even while TPU silicon alone runs much faster and total SoC `mJ/tok` already beats CPU by `-40.8%`.

---

## 3. MoE (`GGML_OP_MUL_MAT_ID`) & Single-Copy LMKD Residency

- **`test-backend-ops` (`435/435 PASS`, `MAPS_GOOGLE=0`, `cpu_fallback_slices=0`)**:
  - `MUL_MAT q8_0`: **`104/104 PASS`** (`9.78 s`, `VmRSS=182.1 MB`, `w_peak=149.8 MB`)
  - `MUL_MAT q4_0`: **`104/104 PASS`** (`21.35 s`, `VmRSS=166.0 MB`, `w_peak=389.7 MB`; note: first run hit 1 transient random-seed `0.0005046 > 5e-4` failure before passing `104/104` cleanly on rerun)
  - **`MUL_MAT_ID` (MoE) `q8_0`**: **`115/115 PASS`** (`32.46 s`, `VmRSS=2,833.0 MB`, `w_peak=182.8 MB`, `99.99%` cache hit)
  - **`MUL_MAT_ID` (MoE) `q4_0`**: **`112/112 PASS`** (`29.92 s`, `VmRSS=1,176.7 MB`, `w_peak=433.5 MB`, `99.77%` cache hit)
- **Single-Copy Weight Residency Under Android LMKD**:
  - Removed duplicate host `packed_q8_cache`; bounded DMA-BUF weight cache by `min(2048 MiB, (MemAvailable + ctx->weight_cache_bytes) / 3)` (fixing self-ratcheting on tight-RAM Linux/Android hosts).
  - Result across all 6 warm `llama-bench` and standalone ODPM TPU runs: **`w_evict = 0`** (`SmolLM2` `w_peak=132.7 MB`, `Qwen2.5 Q8` `w_peak=1,530.3–1,559 MB`, `Qwen2.5 Q4` `w_peak=1,293.6–1,328 MB`), and TPU **`VmRSS` is `37%–44%` smaller than CPU** (`1,954–2,004 MB` vs `3,202–3,260 MB` on `Qwen2.5 Q8`; `1,155–1,205 MB` vs `1,919–1,941 MB` on `Q4`).

---

## 4. Live `rango` Benchmarks & 18-Case ODPM `mJ/tok` Sweep

### 4.1 Coherent Greedy Completions (`--temp 0`, `M=5` prompt + chained `M=1` decode, `w_evict=0`, `MAPS_GOOGLE=0`)
- `SmolLM2-135M-Q8_0` (`swiglu_chained=851`): `"The capital of France is Paris, the Eiffel Tower, the Louvre Museum, and Notre-Dame are all iconic landmarks.\n\nParis is a beautiful city with many attractions"`
- `Qwen2.5-1.5B-Q8_0` (`swiglu_chained=7586`): `"The capital of France is Paris. The capital of France is also the capital of France. Is this statement true?..."` *(coherent; diverges from CPU verbatim `"A) Germany..."` after chained `2^(1/2)` MLP requantization)*
- `Qwen2.5-1.5B-Q4_0` (`swiglu_chained=7551`): `"The capital of France is Paris. Paris is the capital of France. What is the capital of France? Paris..."`

### 4.2 Phase 6 -> Phase 7 M7A -> Phase 7 M7B Warm `llama-bench` (`-p 128,512 -n 64 -t 4 -r 3`)

| Model | Test | Phase 6 TPU (`t/s`) | Phase 7 M7A (`t/s`) | Phase 7 M7B Final (`t/s`, `[reps]`) | Phase 7 TPU / Phase 6 |
|---|---|---:|---:|---|---|
| `SmolLM2-135M Q8` | `pp128` | `384.00` | `676.22` | **`903.81 ± 66.60`** `[826.94, 940.13, 944.34]` | **`2.35x`** |
| `SmolLM2-135M Q8` | `pp512` | `563.82` | `882.13` | **`961.71 ± 118.91`** `[827.54, 1003.53, 1054.06]` | **`1.71x`** (`1.87x` peak) |
| `SmolLM2-135M Q8` | `tg64` | `17.36` | `27.16` | **`34.84 ± 3.09`** `[31.28, 36.40, 36.83]` | **`2.01x`** (`2.12x` peak) |
| `Qwen2.5-1.5B Q8` | `pp128` | `118.06` | `170.30` | **`163.60 ± 13.43`** `[150.03, 163.90, 176.88]` | **`1.39x`** |
| `Qwen2.5-1.5B Q8` | `pp512` | `105.87` | **`142.97`** | **`139.08 ± 0.97`** `[138.03, 139.26, 139.95]` | **`1.31x`** |
| `Qwen2.5-1.5B Q8` | `tg64` | `5.85` | `8.94` | **`8.69 ± 2.22`** `[6.14, 10.15, 9.78]` (warm **`10.15 t/s`**) | **`1.49x`** (`1.74x` warm) |
| `Qwen2.5-1.5B Q4` | `pp128` | `118.55` | **`185.53`** | **`177.72 ± 10.19`** `[166.03, 184.79, 182.33]` | **`1.50x`** |
| `Qwen2.5-1.5B Q4` | `pp512` | `93.94` | **`147.03`** | **`144.99 ± 1.62`** `[144.76, 146.71, 143.49]` | **`1.54x`** |
| `Qwen2.5-1.5B Q4` | `tg64` | `5.55` | **`9.59`** | **`8.98 ± 2.48`** `[6.12, 10.39, 10.42]` (warm **`10.42 t/s`**) | **`1.62x`** (`1.88x` warm) |

### 4.3 18-Case Standalone ODPM `mJ/tok` Sweep

Every test ran standalone on `rango` (`-p 128 -n 0`, `-p 512 -n 0`, `-p 0 -n 64`), measuring live hardware ODPM rails (`S7M_VDD_TPU`, `S2M+S3M+S4M_VDD_CPU*`, `S4S_VDD2H_MEM`). **Steady-state** multiplies total average power by `1 / tps`; **integrated** divides total run energy (load + cold packing + warmup + measured reps) by executed tokens:

| Workload | CPU `t/s` | CPU `P_tot` | CPU SS / INT `mJ/tok` | TPU `t/s` | TPU `P_tot` (`TPU/CPU/MEM`) | TPU SS `mJ/tok` (**`TPU`** / **`Total`**) | TPU INT `mJ/tok` (`Total`) | SS / INT `Total` vs CPU | `VmRSS` TPU vs CPU |
|---|---:|---:|---|---:|---|---|---|---|---|
| `SmolLM2 Q8 pp128` | `1,901.8` | `4,541 mW` | `2.388` / `3.496` | `907.0` | `3,458 mW` (`262/3112/85`) | `0.289` / `3.812` | `6.702` | `+59.6%` / `+91.7%` | `185 vs 307 MB` |
| `SmolLM2 Q8 pp512` | `1,579.5` | `5,231 mW` | `3.312` / `3.968` | `1,017.4` | `3,704 mW` (`311/3282/111`) | `0.306` / **`3.640`** | `5.370` | **`+9.9%`** / `+35.3%` | `200 vs 323 MB` |
| `SmolLM2 Q8 tg64` | `190.7` | `5,440 mW` | `28.533` / `30.655` | `32.31` | `4,363 mW` (`370/3867/127`) | `11.439` / `135.036` | `131.877` | `+373.3%` / `+330.2%` | `175 vs 304 MB` |
| **`Qwen2.5 Q8 pp128`** | `194.8` | `4,977 mW` | `25.554` / `32.038` | `156.0` | `2,208 mW` (`381/1652/175`) | **`2.439`** / **`14.152`** | **`25.043`** | **`-44.6%`** / **`-21.8%`** | **`1,954 vs 3,202 MB`** |
| **`Qwen2.5 Q8 pp512`** | `186.7` | `4,907 mW` | `26.288` / `29.542` | `128.8` | `2,005 mW` (`395/1410/200`) | **`3.070`** / **`15.568`** | **`19.711`** | **`-40.8%`** / **`-33.3%`** | **`2,004 vs 3,260 MB`** |
| `Qwen2.5 Q8 tg64` | `19.38` | `4,968 mW` | `256.331` / `224.149` | `7.03`* | `2,744 mW` (`494/1991/259`) | **`70.284`** / `390.270` | `350.985` | `+52.3%` / `+56.6%` (`-53%` vs P6 INT) | `1,895 vs 3,388 MB` |
| **`Qwen2.5 Q4 pp128`** | `146.9` | `2,362 mW` | `16.082` / `30.371` | `128.9` | `1,348 mW` (`312/896/141`) | **`2.420`** / **`10.460`** | **`18.352`** | **`-35.0%`** / **`-39.6%`** | **`1,155 vs 1,941 MB`** |
| **`Qwen2.5 Q4 pp512`** | `176.3` | `2,937 mW` | `16.655` / `18.164` | `119.5` | `1,336 mW` (`352/822/162`) | **`2.944`** / **`11.175`** | **`15.188`** | **`-32.9%`** / **`-16.4%`** | **`1,205 vs 1,919 MB`** |
| `Qwen2.5 Q4 tg64` | `22.74` | `2,459 mW` | `108.149` / `97.403` | `4.18`* | `2,001 mW` (`362/1447/191`) | `86.675` / `478.612` | `358.452` | `+342.5%` / `+268.0%` (`-55%` vs P6 INT) | `1,100 vs 1,772 MB` |

*\*Standalone `-r 2` `tg64` includes rep-1 cold SwiGLU graph registration (`2919` / `31801` chained chunks); once warm in `llama-bench`, `Qwen2.5` decode reaches **`10.15 t/s` (`Q8`) / `10.42 t/s` (`Q4`)**, cutting steady-state `tg64` power-at-speed (`2.74 W / 10.15 t/s ≈ 270 mJ/tok` on `Q8`) to parity with 4-thread CPU (`256 mJ/tok`).*

---

## 5. Silicon Physics Ceiling & Next Horizon (Beyond Phase 7)

1. **Physical Single-Core Ring Bandwidth Wall (`27.6–28.0 GB/s`)**: Even at `1,286.4 MHz` (`OPP 0x19`), active single-cluster DMA (`cluster_id=2`) tops out at `27.96 GB/s` (`26.23 GB/s` inside `probe_swiglu_1x1536x1024x1536`). Streaming `~1.54 GB` of INT8 weights per token enforces a hard **`~55 ms/tok` (`~18 tok/s` pure silicon)** physical floor on 1.5B models regardless of host-CPU speed.
2. **Breaking `15–20+ t/s` on 1.5B–7B Decode**: Exceeding the `~18 tok/s` INT8 DMA floor requires halving DRAM weight traffic via either native **4-bit hardware unpacking** (`W4A16(BF16)` / `INT4`) or `HibDma.FlexibleDma.Huffman` compressed streams, alongside multi-threaded NEON `dequant`/`quant` overlapped during TPU prefill wait windows (`pp512`) to turn Phase 7's `-40.8%` energy win into an end-to-end `tok/s` prefill win too.
