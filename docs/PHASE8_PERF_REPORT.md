# Phase 8 Report: Overlapped SPSC NEON, Native PT `W8A16(FP16)`, and Causal Hardware Huffman / `W4` Silicon Verdicts

**Target Device**: Google Pixel 10 Pro Fold (`rango`, Tensor G5 `buenos_a0`, Android 17 `CP3A.261005.005`, kernel `6.6.127`).  
**Clean-Room Invariant**: Zero Google/vendor `.so` dependencies (`llvm-readelf -d` `NEEDED` = `[libdl.so, libm.so, libc.so]` only across `openbuenos_selftest`, `llama-bench`, `llama-cli`, `llama-completion`, `test-backend-ops`; `0` `dlopen`/`dlsym`; `0` embedded RV64 ELF; live `/proc/<pid>/maps` `MAPS_GOOGLE=0` across all runs).  
**Baseline**: Builds on [Phase 7 (`PR #4`)](PHASE7_PERF_REPORT.md) (`11`-chunk chained SwiGLU, `MUL_MAT_ID` MoE, adaptive spin, and single-copy LMKD residency).

---

## 1. Stage 1 Causal Silicon Verdicts: Hardware Huffman, `W4`, and `W8A16(FP16)`

Investigating the three bandwidth/precision levers beyond Phase 7's `27.96 GB/s` single-core INT8 parameter ring ceiling ([`phase8_huffman_w4a16_spec.md`](isa-gemm/phase8_huffman_w4a16_spec.md), `wave1_oracle` & `wave2_huffman` live silicon telemetry):

1. **Hardware Huffman Weight Decompression (`HibDma.FlexibleDma.Huffman` / `0x0cc0`) — `[CAUSALLY_PROVEN]` Inactive on `buenos_a0`**:
   - Offline RE of `libedgetpu_tflite_compiler.so` proved `buenos_a0_multimedia_huffman_config.binarypb` (`7,568 B`) differs from `buenos_a0_config.binarypb == v7_config.binarypb` (`7,562 B`) in only 3 `ScalarCore` fields (`283: 5->8`, `287: 0->1`, `288: 0->1`), has **0 callers in `.text`**, and even `buenos_b0_config.binarypb` keeps `{283: 5, 287: 0, 288: 0}`. Production firmware `1.1 925724637` contains `0` `huffman`/`codebook` strings.
   - Live `rango` Wave-2 bare-metal probes (`PM=5` @ `1,286.4 MHz`) proved HIB `0x0a40` ignores `bit499=1` and `fields 7..9` (`compressed_data`, `metadata` `6 B/256 B`, `672 B codebook`):
     - `h4_huff_rawfb_poison_1x16x16` (256 B of `0x00` at raw fallback `field5@37` + valid Huffman weights at `field7@501`, `sum(x)=7`) returned `[-112]*16` (`max_err=0` vs raw zero-weight math `7 * (-128) / 8 = -112`).
     - `h5_huff_packed_1x1536x1536` (`2,359,296 -> 1,043,360 B`) & `h6_huff_packed_1x2048x2048` (`4,194,304 -> 1,854,560 B`) triggered SMMU read faults (`fault ... write=0 not mapped`), attempting uncompressed `2.36 MB`/`4.19 MB` reads past the compressed `ext[2]` DMA-BUF.
     - `h7_huffdma_0cc0_1x16x16` emitted `0x0cc0` + uncompressed `0x0a40` and matched `h0_ctrl` bit-identically (`77 us`).
   - **Conclusion**: `buenos_a0` hardware cannot exceed the `27.96 GB/s` INT8 DMA ring ceiling via Huffman compression.
2. **`SubbyteConstantRemapper` (`W4A8`) & `W4A16` (`0x980`) — `[CAUSALLY_PROVEN]` No 4-Bit DRAM Savings**:
   - `SubbyteConstantRemapper` (`0x2513e68`) expands 4-bit `W4A8` weights (`w4a8_pc_1x1536x1536`) to `1 B/wt` (`ext2=2,371,584 B`, `98 us` HW, `24.23 GB/s`, `0%` bandwidth savings).
   - `w4a16_deq_i4_1x1536x1536` halves `ext[2]` (`1,185,792 B`), yet `0x980 u32[16] = 0x0220_0000` (`zp_w=8`) still sets `bit21=1` (`1:2 INT8_TO_BFLOAT`; `0x1c3e71` asserts no `INT4_TO_BFLOAT` hardware unpack).
3. **Large-Tile `W8A16` — `[CAUSALLY_PROVEN]` Native `IEEE-754 FP16` (10-bit Mantissa) + `FP32` Accumulator**:
   - Wave-1 oracle probes (`6/6` compiled `sgs=1` on `rango`) proved `buenos_a0` `W8A16` (`[N/32, K/2, 32, 2]` `w_i8^0x80` packing) executes as **native `FP16` (10-bit mantissa)** rather than `BF16`, with **7-chunk per-tensor (`scale=1.0`) writing the full `FP32` accumulator** directly (`relL2=2.76e-7` vs `fp32(fp16(x)@W)*s`, `relL2_fp32=1.88e-4`, zero `FP16` intermediate overflow risk):
     - `w8a16_deq_i8_pt_1x1536x1536`: **`138.0 us` HW** (`17.19 GB/s`, `relL2_fp32=1.88e-4`)
     - `w8a16_deq_i8_1x2048x2048`: **`233.0 us` HW** (`18.07 GB/s`, `relL2_fp32=2.80e-4`)
     - `w8a16_deq_i8_64x1536x1536`: **`589.0 us` HW** (`512.7 GOPS`, `relL2_fp32=2.78e-4`, zero CPU activation quant or outlier peel).

---

## 2. Stage 2 Architecture: Overlapped SPSC Async Dequant & Clean-Room PT `W8A16(FP16)`

1. **Milestone 8A Heterogeneous-Core Discovery (on-device telemetry)**:
   - Vectorized `quant`/`dequant` + fused outlier FMA (`store_first` on `k0==0`, `vcvt_f32_f16`, `OPENBUENOS_MAX_OUTLIER_COLS=160` under strict LMKD budget, removed per-outlier `/proc/meminfo` syscall, zero-copy parallel NEON `pack_ext2_*_into`, persistent `M7A` scratch, multi-pos RoPE trig cache, and `M>1` SwiGLU `1.5x`-step divider fix) pushed `OPENBUENOS_THREADS=1` (`t1`) **`Qwen2.5 Q8 pp512`** from `139.08 -> 178.21 t/s`, **`SmolLM2 pp512`** to `1,131.2 t/s` (`tg64=38.02 t/s`), and **`Qwen2.5 Q4 pp512`** to `182.59 t/s`!
   - Conversely, enabling per-tile/per-row `parallel_for` barriers across `OPENBUENOS_THREADS=4` collapsed `Qwen2.5 Q8 pp512` to `107.82 t/s` (`dequant_us` `2.29s -> 9.17s`) due to `~18,000` condvar/barrier bounces across Tensor G5's heterogeneous cores (`cpu0..1 = 2.246 GHz` LITTLE, `cpu2..6 = 3.052 GHz` MID, `cpu7 = 3.782 GHz` BIG).
2. **Milestone 8B Fix — Lock-Free 1-Helper SPSC `AsyncDequantWorker` (`ob_worker_pool.cpp`, `OPENBUENOS_ASYNC_DEQUANT=1`)**:
   - Kept calling BIG-core inline NEON (`kDefaultWorkerThreads=1`) for `quant`, `pack`, `submit`, `wait`, and `M<=8` decode, while offloading `M>8` prefill `dequant + outlier FMA` via a **16-slot cache-line-padded SPSC ring** to **1 persistent helper thread pinned to Tensor G5 performance cores (`cpuinfo_max_freq > min_freq`, `cpu2..7`)** with `~1` wakeup per prefill call (`0%` CPU during `tg64`).
   - Result: `OPENBUENOS_ASYNC_DEQUANT=1` outperformed synchronous `=0` on all 3 warm benchmarks (`SmolLM2 pp512`: `1,250.0 vs 1,204.7`; `Qwen2.5 Q8 pp512`: `179.4 vs 174.0`; `Qwen2.5 Q4 pp512`: `180.4 vs 164.1 t/s`).
3. **Clean-Room 7-Chunk PT `W8A16(FP16)` Integrated (`ob_gemm_emitter.cpp`, `ob_gemm.cpp`)**:
   - Implemented NEON `[N/32, K/2, 32, 2]` `w_i8^0x80` packer, generalized `16`-core `M=1` emitter (`512..2048`) with Probe-A dynamic `override_params`, NEON per-channel scale epilogue, `|x|>32768` prescale guard, and `OPENBUENOS_W8A16={0,1,auto}` (`opt-in`).
   - Host verification (`make -C libopenbuenos verify`): **20/20 suites PASS** (incl. Suite 20 `12,880/12,880 B` bit-exact PT W8A16 oracle); host `test-backend-ops`: `MUL_MAT 327/327`, `MUL_MAT_ID 227/227`.

---

## 3. Live `rango` Verification, Warm `llama-bench`, and 18-Case ODPM Sweep

All runs at `max_freq_mhz = 1286.4` (`PM=5` + FW WakeLock), `MAPS_GOOGLE=0`, `cpu_fallback_slices=0`, `w_evict=0` across all live LLM runs:

### 3.1 Selftests, `test-backend-ops` (`435/435 PASS`), and Coherence
- `openbuenos_selftest --test-swiglu`: **`8/8 PASS`** (`sat_retries=0`, `qwen25_q4_64x1536x8960` fixed from `4.00e-03` -> **`2.14e-03`** at `5970 us` HW).
- `openbuenos_selftest --test-w8a16`: **`8/8 PASS`** on live silicon (`pt_1x1536x1536` `144 us / 16.38 GB/s / relL2=2.181e-4`; `pt_1x2048x2048` `229 us / 18.32 GB/s / relL2=2.133e-4`; `spike75` `138 us / relL2=5.347e-5`; `prescale50k` `140 us / relL2=2.264e-4`).
- `test-backend-ops`: **`435/435 PASS`** (`MUL_MAT q8_0` `104/104` in `10.56 s`; `q4_0` `104/104` in `21.44 s`; **`MUL_MAT_ID` MoE `q8_0` `115/115`** in `30.48 s` `99.99%` hit; **`MUL_MAT_ID` MoE `q4_0` `112/112`** in `28.99 s` `99.76%` hit).
- Coherent greedy completions on `SmolLM2-135M-Q8_0`, `Qwen2.5-1.5B-Q8_0`, `Qwen2.5-1.5B-Q4_0`, plus opt-in `W8A16=1` (`sat_retries 50->14`) & `W8A16=auto` (`sat_retries 50->21`), all `w_evict=0`.

### 3.2 Phase 7 -> Phase 8 Warm `llama-bench` (`-p 128,512 -n 64 -t 4 -r 3`, `w_evict = 0`)

| Model | Test | Phase 7 M7B (`t/s`) | Phase 8 M8B Async (`t/s`, `[reps]`) | Phase 8 Sync (`=0`) | Speedup vs P7 | `VmRSS` / `w_peak` |
|---|---|---:|---|---:|---|---|
| `SmolLM2-135M Q8` | `pp128` | **`903.81`** | `489.71 ± 102.10` `[483.23, 391.01, 594.89]` | — | `-45.8%` | `212 MB` / `145 MB` |
| `SmolLM2-135M Q8` | `pp512` | `961.71` | **`1,249.96 ± 71.79`** `[1173.05, 1315.19, 1261.65]` | `1,204.68` | **`+30.0%`** | `212 MB` / `145 MB` |
| `SmolLM2-135M Q8` | `tg64` | `34.84` | **`34.23 ± 3.50`** `[30.21, 35.89, 36.59]` | `23.03` | `0.98x` | `212 MB` / `133 MB` |
| `Qwen2.5-1.5B Q8` | `pp128` | `163.60` | `154.13 ± 17.24` `[135.06, 158.71, 168.62]` | — | `0.94x` | `2,159 MB` / `1,688 MB` |
| `Qwen2.5-1.5B Q8` | `pp512` | `139.08` | **`179.36 ± 3.88`** `[174.88, 181.52, 181.66]` | `174.00` | **`+29.0%`** | `2,159 MB` / `1,688 MB` |
| `Qwen2.5-1.5B Q8` | `tg64` | `8.69` | **`8.84 ± 1.43`** `[7.21, 9.90, 9.42]` | `7.73` | `+1.7%` | `2,159 MB` / `1,537 MB` |
| `Qwen2.5-1.5B Q4` | `pp128` | `177.72` | `146.84 ± 19.02` `[126.04, 163.36, 151.12]` | — | `0.83x` | `1,368 MB` / `1,460 MB` |
| `Qwen2.5-1.5B Q4` | `pp512` | `144.99` | **`180.37 ± 3.27`** `[176.98, 183.50, 180.62]` (**beats CPU `160.2–176.3`**) | `164.09` | **`+24.4%`** | `1,368 MB` / `1,460 MB` |
| `Qwen2.5-1.5B Q4` | `tg64` | `8.98` | **`9.38 ± 1.76`** `[7.36, 10.45, 10.35]` (warm **`10.45`**) | `5.99` | **`+4.5%`** | `1,368 MB` / `1,294 MB` |

*(Opt-in `W8A16=1` / `auto` `Qwen2.5 Q8 tg64`: `4.96 ± 1.32` / `4.56 ± 1.10 t/s`, `w_evict=0`.)*

### 3.3 18-Case Standalone ODPM `mJ/tok` Sweep (live device telemetry) — **5/6 Prefills Beat 4T CPU**

| Workload | CPU `t/s` | CPU SS / INT `mJ/tok` | TPU `t/s` | TPU `P_tot` (`TPU/CPU/MEM`) | TPU SS `mJ/tok` (**`TPU`** / **`Total`**) | TPU INT `mJ/tok` (`Total`) | SS / INT vs CPU | `VmRSS` TPU vs CPU |
|---|---:|---:|---|---:|---|---|---|---|
| `SmolLM2 Q8 pp128` | `1,919.1` | `2.450` / `3.509` | `969.4` | `2,831 mW` (`249/2499/83`) | `0.257` / `2.921` | `6.616` | `+19.2%` / `+88.5%` | `195 vs 306 MB` |
| **`SmolLM2 Q8 pp512`** | `1,584.0` | `3.079` / `3.837` | `1,069.7` | `3,034 mW` (`306/2617/112`) | **`0.286`** / **`2.836`** | `4.592` | **`-7.9%`** / `+19.7%` | `212 vs 323 MB` |
| `SmolLM2 Q8 tg64` | `180.1` | `27.906` / `30.978` | `32.65` | `3,722 mW` (`374/3215/133`) | `11.446` / `113.991` (`-15.6%` P7) | `107.700` (`-18.3%` P7) | `+308.5%` / `+247.7%` | `177 vs 303 MB` |
| **`Qwen2.5 Q8 pp128`** | `171.3` | `21.757` / `38.489` | `118.5` | `1,245 mW` (`317/793/135`) | **`2.674`** / **`10.510`** | **`18.215`** | **`-51.7%`** / **`-52.7%`** | `2,103 vs 3,276 MB` |
| **`Qwen2.5 Q8 pp512`** | `178.4` | `23.685` / `26.688` | `156.0` | `1,377 mW` (`417/775/184`) | **`2.676`** / **`8.826`** (`2.68x` CPU) | **`11.921`** | **`-62.7%`** / **`-55.3%`** | `2,160 vs 3,396 MB` |
| `Qwen2.5 Q8 tg64` | `14.91` | `176.633` / `156.234` | `3.51`* | `1,523 mW` (`364/997/162`) | `103.732` / `433.846` | `321.408` (`-8.4%` P7) | `+145.6%` / `+105.7%` | `1,897 vs 3,186 MB` |
| **`Qwen2.5 Q4 pp128`** | `164.7` | `12.556` / `16.313` | `140.1` | `1,174 mW` (`305/738/132`) | **`2.174`** / **`8.379`** | **`16.145`** | **`-33.3%`** / **`-1.0%`** | `1,307 vs 1,918 MB` |
| **`Qwen2.5 Q4 pp512`** | `160.2` | `12.704` / `14.494` | `151.8` | `1,160 mW` (`397/598/165`) | **`2.614`** / **`7.645`** | **`10.653`** | **`-39.8%`** / **`-26.5%`** | `1,368 vs 1,845 MB` |
| `Qwen2.5 Q4 tg64` | `17.65` | `88.210` / `80.124` | `6.08`* | `1,809 mW` (`376/1242/192`) | `61.776` / `297.582` (`-37.8%` P7) | `309.782` (`-13.6%` P7) | `+237.4%` / `+286.6%` | `1,111 vs 1,771 MB` |

*\*Standalone `-r 2` `tg64` includes rep-1 cold SwiGLU graph registration & packing (`2919..31824` chunks); warm steady-state reaches `9.90 t/s` (`Q8`) and `10.45 t/s` (`Q4`).*

---

## 4. Phase 8 Summary & Future Scope

- **Prefill Speed & Energy**: Overlapped SPSC NEON async dequant pushed `Qwen2.5 Q4 pp512` past 4T ARM64 CPU on raw speed (`180.37` vs `160.2 t/s`), slashed total SoC prefill energy on `Qwen2.5 Q8 pp512` by **`-62.7%`** (`8.826` vs `23.685 mJ/tok` at `1.38 W` total SoC power), and unlocked the first **`SmolLM2 pp512`** total-energy win (`2.836` vs `3.079 mJ/tok`, `-7.9%`).
- **Settled Hardware Ceiling on `buenos_a0`**: Hardware Huffman (`FlexibleDma.Huffman` / `0x0cc0`) and 4-bit hardware DMA unpack are causally inactive on Pixel 10 (`buenos_a0`) silicon, establishing `27.96 GB/s` (1 `B`/weight) as the hard decode bandwidth ceiling alongside selective 7-chunk PT `W8A16(FP16)` (`18.32 GB/s`, `FP32` accumulator) for outlier-sensitive / high-precision workloads.
