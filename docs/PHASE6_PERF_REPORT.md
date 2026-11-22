# Phase 6 Performance Report: Tensor G5 DVFS Unlock, Batched Prefill, Chaining & Energy Analysis

**Target Device**: Google Pixel 10 Pro Fold (`rango`, Tensor G5 `buenos_a0`, Android 17 `CP3A.261005.005`, kernel `6.6.127`).  
**Clean-Room Invariant**: Zero Google/vendor `.so` dependencies (`[libdl.so, libm.so, libc.so]` only on all stripped Android ARM64 ELFs; `0` `dlopen`/`dlsym`; `MAPS_GOOGLE=0` live via `/proc/<pid>/maps`).

---

## 1. Root Cause of Phase 5B Clock Cap & Bare-Metal Unlock (`38.4 -> 1,286.4 MHz`)

- **Wire-Protocol Shift Bug** ([`dvfs_power_states.md`](baremetal-vii/dvfs_power_states.md)): Live wire captures and binary analysis of `VII_CMD #2` (`struct litebuf_vii_execute`) proved a 1-byte shift at `+0x1c..+0x1f`:
  - `+0x1c`: `fw_priority` (lower = higher priority; `0x01` in `PM=5`)
  - `+0x1d`: `tpu_core_power_state` (`0x01, 0x03..0x19`, where `0x03` = `38.4 MHz` sleep clock and **`0x19` = `1,286.4 MHz` peak OPP** — all of Phase 5B ran at `38.4 MHz`)
  - `+0x1e`: `memory_power_state` (`0x01..0x15`, `0x15` max)
  - `+0x1f`: `qos_class` (`{0, 1, 2, 4, 5}`)
  - `+0x20`: `cluster_id` (`0x02` single cluster on `rango`)
- **Litebuf Firmware WakeLock (`AcquireWakeLock`, `union_type=4`, `bitset=0x04`)**: Sending `AcquireWakeLock` on context init / `perf_mode >= 1` (`ReleaseWakeLock` `union_type=5` on destroy or `PM=0`) eliminates per-call power-gate churn (`buenos.ko` `edgetpu_soc_handle_reverse_kci` returns `-EOPNOTSUPP`, so all DVFS/GMC votes flow through Litebuf VII).

### Bare-Metal Single-Op Silicon Scaling (`rango`, `max_abs_err=0`, zero Google `.so`)

| Kernel / Shape | Config (`fw_wl`, `prio,core,mem,qos`) | `hw_us` | `wall_ms` | Weight BW | Throughput | TPU Power | Speedup |
|---|---|---:|---:|---:|---:|---:|---|
| `1x2048x2048` (`4.0 MiB`) | `PM=0` (`0`, `0x0b,0x03,0x01,1`, `38.4 MHz`) | `5,869` | `6.594` | `0.71 GB/s` | `1.4 GOPS` | `26.1 mW` | baseline |
| `1x2048x2048` (`4.0 MiB`) | `PM=5` no-WL (`0`, `0x0b,0x19,0x0d,1`) | `341` | `1.184` | `12.30 GB/s` | `24.6 GOPS` | `203.1 mW` | `17.2x` hw / `5.6x` wall |
| **`1x2048x2048`** (`4.0 MiB`) | **`PM=5` + FW WL** (`1`, `0x01,0x19,0x15,2`) | **`152`** | **`0.482`** | **`27.59 GB/s`** | **`55.2 GOPS`** | **`114.2 mW`** | **`38.6x` hw / `13.7x` wall** |
| `64x1536x1536` (`151M MAC`) | `PM=0` (`0`, `0x0b,0x03,0x01,1`, `38.4 MHz`) | `10,101` | `12.190` | `0.23 GB/s` | `29.8 GOPS` | `41.4 mW` | baseline |
| **`64x1536x1536`** (`151M MAC`) | **`PM=5` + FW WL** (`1`, `0x00,0x19,0x0d,1`) | **`307`** | **`0.674`** | **`7.69 GB/s`** | **`983.6 GOPS`** | **`177.2 mW`** | **`32.9x` hw / `18.1x` wall (`5.55 TOPS/W`)** |

In `openbuenos_selftest`, `PM=0 -> PM=5` on `1x2048x2048` measured `4075 -> 150 us hw` (`27.2x`, `27.96 GB/s`) and `6724 -> 395 us wall` (`17.0x`); `64x1536x1536` INT8 hit **`304 us hw` (`993.4 GOPS`)**.

---

## 2. Runtime & ISA Engineering (`libopenbuenos` + `ggml-edgetpu` + Chaining)

1. **`M=64` Batched Prefill Large-Tile ISA** ([`ob_gemm_emitter.cpp`](../libopenbuenos/src/ob_gemm_emitter.cpp), [`ob_gemm.cpp`](../libopenbuenos/src/ob_gemm.cpp)):
   - Clean-room C++17 synthesizer for `M=64, K in {512,576,1024,1536}, N in [128..1536]` step `32` (`K+N <= 3072`, separated from `M=1` weights via `pack_fmt`), cutting `Qwen2.5` `pp128` `Q`/`O` from `144` tiles down to **`2` tiles/proj** and `SmolLM2` `pp64` to `7` tiles/layer.
   - Host test parity (`openbuenos_host_test`): **16/16 suites PASS**, **92/92 bit-exact** (`81/81` + `6/6 lt_1x` + `5/5 lt_64x`), `verify-android` clean.
2. **Native `Q4_0` + Shared-`A` Batched Async Ring** ([`openbuenos.h`](../libopenbuenos/include/openbuenos.h), [`ggml-edgetpu.cpp`](../llama-cpp-edgetpu/ggml-edgetpu.cpp)):
   - Added `ob_gemm_matmul_f32_quant_batch` supporting mixed `Q4_0`/`Q8_0` weights directly from mmap'd GGUF pointers into `O(1)` strided-hash `ext[2]` DMA-BUF LRU caches (`<= 4096` weight entries + `<= 256` pooled DMA-BUFs to prevent `RLIMIT_NOFILE=32768` `EMFILE` on `M > 1` prefill).
   - Shared-`A` batching pipelines sibling `Q/K/V` and `up/gate` projections concurrently across 16 async credits (`q4_qkv_1x576` `3`-branch `Q4_0`: `122 us hw / 487 us wall`; `q4_gate_up_64x576x1536`: `282 us hw / 2.31 ms wall`; `q4_prefill_64x1536x1536`: `307 us hw / 1.06 ms wall`), plus post-`MUL_MAT` 1D bias `ADD` inside unified `EDGETPU` graph splits.
   - Retry multiplier schedule tuned to `{1.25x, 1.5x, 2.0x, 4.0x, 4.0x}` (`1.0x` initial `scale_d`), resolving all marginal `test-backend-ops` `src1=[0,1]` and batched `N` NMSE edge cases.
3. **Zero-DRAM Multi-Op `0xa40` Chaining & `W8A16(BF16)` Spec** ([`chained_fusion_and_w8a16_spec.md`](isa-gemm/chained_fusion_and_w8a16_spec.md)):
   - Decoded multi-op `ext[2]` concatenation (`2*S+5` chunks, DIVE `Add`/`PutBits`, `0x940`/`0x980` param ring stepping, `0x100` binary `MUL` in `fuse_swiglu_64`, `0x600`/`0x540`/`0x6c0` concat in `fuse_qkv_64`).
   - Verified `3/3` chained silicon probes (`fuse_fc64_silu_fc64` `60.8 us`, `fuse_swiglu_64` `65.4 us`, `fuse_qkv_64` `72.8 us`) and `8/8 ext[2]` slices **100% bit-exact** (`rango bm==ref err=0`) via [`tools/verify_chained_fusion.py`](../tools/verify_chained_fusion.py).

---

## 3. Live `rango` Operator & LLM Throughput Benchmarks

### 3.1 `test-backend-ops` (`PM=5` @ `1,286.4 MHz`, `MAPS_GOOGLE=0`, `cpu_fallback_slices=0`)
- **`type_a=q8_0,type_b=f32`**: **`104/104` supported ops PASS** (`12.1 s` total vs `485.76 s` in Phase 5B; `hw_us=0.92 s` = **`47.5x` faster silicon**; `99.87%` hit rate; `31.9°C`; `175.1 mW` TPU).
- **`type_a=q4_0,type_b=f32`**: **`104/104` supported ops PASS** (`23.5 s` total; `hw_us=0.84 s` = **`87.9x` faster silicon**; `99.82%` hit rate; `31.8°C`; `154.4 mW` TPU).

### 3.2 Phase 5B (`38.4 MHz`) vs Phase 6 (`1,286.4 MHz` + `lt_64x*`) vs ARM64 CPU `-t 4` (`llama-bench` `-p 128,512 -n 64 -r 3`)

| Model (`rango`) | Test | Phase 5B TPU (`t/s`) | Phase 6 TPU Warm (`t/s`) | Pure Silicon (`hw_us` `t/s`) | ARM64 CPU `-t 4` (`t/s`) | Speedup vs 5B | Peak TPU Temp |
|---|---|---:|---:|---:|---:|---|---|
| `SmolLM2-135M-Q8_0` | `pp128` | `0.47` | **`384.00 ± 16.59`** | **`2,813`** | `1,928.45` | **`817.0x`** | `56.0°C` |
| `SmolLM2-135M-Q8_0` | `pp512` | — | **`563.82 ± 48.40`** | **`3,007`** (**`1.80x` CPU**) | `1,673.33` | — | `56.0°C` |
| `SmolLM2-135M-Q8_0` | `tg64` | `1.09` | **`17.36 ± 2.74`** | `77` | `173.88` | **`15.9x`** | `56.0°C` |
| `Qwen2.5-1.5B-Q8_0` | `pp128` | — | **`118.06 ± 8.39`** | **`354`** | `211.75` | batched unlocked | `62.0°C` |
| `Qwen2.5-1.5B-Q8_0` | `pp512` | — | **`105.87 ± 1.65`** | **`363`** (**`1.77x` CPU**) | `205.50` | batched unlocked | `62.0°C` |
| `Qwen2.5-1.5B-Q8_0` | `tg64` | `0.27` (`tg16`) | **`5.85 ± 0.92`** | `~13.5` | `19.36` | **`21.7x`** | `62.0°C` |
| `Qwen2.5-1.5B-Q4_0` | `pp128` | — | **`118.55 ± 0.94`** | `~364` | `228.14` | batched unlocked | `59.0°C` |
| `Qwen2.5-1.5B-Q4_0` | `pp512` | — | **`93.94 ± 4.44`** | `~367` | `218.63` | batched unlocked | `59.0°C` |
| `Qwen2.5-1.5B-Q4_0` | `tg64` | `0.16` (`tg16`) | **`5.55 ± 3.05`** (`7.16` cold) | `~15.7` | `30.75` | **`34.7x–44.8x`** | `59.0°C` (`64°C` cold) |

Greedy output correctness (`--temp 0 --seed 42`, batched prefill enabled): **`Qwen2.5-1.5B-Q8_0` matched CPU verbatim 22/22 tokens** (`"The capital of France is Paris. The capital of France is also the capital of which other country?\nA) Germany\nB) Italy\nC) Spain\nD) United Kingdom"`, `99.58%` cache hit, `cpu_fallback_slices=0`, `MAPS_GOOGLE=0`).

---

## 4. Apples-to-Apples Power & Energy Sweep (`mJ/tok` via Live ODPM Rails)

Measured across 18 standalone single-test runs (`-p 128 -n 0`, `-p 512 -n 0`, `-p 0 -n 64`) reading hardware energy accumulators `S7M_VDD_TPU`, `S2M+S3M+S4M_VDD_CPU*`, and `S4S_VDD2H_MEM` on `rango`:

| Workload | CPU `t/s` | CPU Total Power | CPU SS `mJ/tok` (`Total`) | TPU `t/s` | TPU Rail Power (`TPU / CPU / MEM`) | TPU SS `mJ/tok` (**`TPU`** / `CPU` / `MEM` / **`Total`**) | TPU Integrated `mJ/tok` vs CPU |
|---|---:|---:|---:|---:|---|---|---|
| `SmolLM2 Q8 pp128` | `1979.51` | `5963 mW` | `3.01` | `407.05` | `241.8 / 4316.7 / 75.1 mW` | `0.59 / 10.61 / 0.18 / 11.38` | `16.49` vs `4.11` |
| `SmolLM2 Q8 pp512` | `1647.07` | `6029 mW` | `3.66` | `591.87` | `289.9 / 4178.6 / 98.4 mW` | **`0.49`** `/ 7.06 / 0.17 / 7.72` | `10.12` vs `4.48` |
| `SmolLM2 Q8 tg64` | `191.04` | `5707 mW` | `29.88` | `17.90` | `349.6 / 4354.6 / 120.3 mW` | `19.53 / 243.28 / 6.72 / 269.53` | `325.29` vs `35.28` |
| **`Qwen2.5 Q8 pp128`** | `204.21` | `5925 mW` | `29.01` | `115.06` | `333.5 / 2397.7 / 143.0 mW` | `2.90 / 20.84 / 1.24 /` **`24.98`** (**`-13.9%`**) | `50.01` vs `36.73` |
| **`Qwen2.5 Q8 pp512`** | `202.63` | `5887 mW` | `29.05` | `93.66` | `332.6 / 1655.0 / 147.3 mW` | **`3.55`** `/ 17.67 / 1.57 /` **`22.79`** (**`-21.5%`**) | **`31.22` vs `32.72`** (**`-4.6%`**) |
| `Qwen2.5 Q8 tg64` | `19.31` | `5943 mW` | `307.78` | `6.14` | `480.5 / 2477.3 / 241.3 mW` | **`78.26`** `/ 403.48 / 39.30 / 521.04` | `745.75` vs `362.50` |
| **`Qwen2.5 Q4 pp128`** | `204.29` | `3568 mW` | `17.46` | `70.82` | `244.8 / 822.5 / 98.4 mW` | `3.46 / 11.61 / 1.39 /` **`16.46`** (**`-5.7%`**) | `31.18` vs `21.33` |
| `Qwen2.5 Q4 pp512` | `174.14` | `3107 mW` | `17.84` | `69.78` | `282.9 / 873.8 / 116.7 mW` | **`4.05`** `/ 12.52 / 1.67 / 18.25` | `22.90` vs `19.21` |
| `Qwen2.5 Q4 tg64` | `23.99` | `2735 mW` | `113.99` | `3.18`* | `309.8 / 1521.3 / 141.1 mW` | `97.43 / 478.39 / 44.36 / 620.17` | `797.16` vs `134.06` |

*\*CPU big-core thermal throttling reduced CPU clocks during the final `qwen_q4` sweep (`P_CPU 2.2–3.4 W` vs `5.2–5.7 W` on `q8`), confirming host-side CPU work bottlenecks `tg64`.*

**Energy Conclusion**: **`Qwen2.5-1.5B-Q8_0` `pp512` already beats 4-thread ARM64 CPU in total system `mJ/tok`** (`-21.5%` steady-state, `-4.6%` integrated including model load + warmup). The **TPU rail itself uses `3.4x–8.0x` less energy than CPU** (`3.55 vs 28.26 mJ/tok` on `Qwen2.5 Q8 pp512`; `78.26 vs 266.89` on `tg64`), while **77–90% of system energy during TPU runs is spent in host CPU** (`1.5–4.4 W`).

---

## 5. Exact Next Bottleneck & Phase 7 Roadmap

Pure Tensor G5 silicon (`hw_us`) is already **`1.77x–1.80x` faster than 4-thread ARM64 CPU on `pp512`**. Closing the remaining `2.3x–5.3x` `wall_us / hw_us` host-CPU gap requires:
1. **Eliminate Host-Side CPU & DMA-BUF Churn**: Fix `pool_hits=0 / pool_misses=180..332` via persistent tensor-bound activation DMA-BUFs across graph evals and move `a_mk` quantization / `c_mn` FP32 accumulation onto ARM64 NEON + multi-threaded worker dispatch (projected to push `Qwen2.5-1.5B` `pp512` well past `250–300 t/s` end-to-end vs CPU `202 t/s` and slash total SoC `mJ/tok` >2x).
2. **Integrate Verified `0xa40` Multi-Op Chaining + Native `W8A16(BF16)`**: Wire `fuse_swiglu_64` (`0x640 + 0x100 MUL`) and `fuse_qkv_64` / horizontal concat (`[W_Q; W_K; W_V]`, `[W_gate; W_up]`) directly into `ob_gemm_emitter.cpp` & `ggml-edgetpu.cpp` to cut decode VII round-trips ~3–5x.
3. **MoE & 7B–14B Readiness**: Implement `GGML_OP_MUL_MAT_ID` routed expert slice dispatch and single-copy weight residency for 16 GB Pixel targets under Android LMKD.
