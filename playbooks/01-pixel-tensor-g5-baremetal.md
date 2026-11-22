# Playbook 01: Pixel Tensor G5 (`buenos_a0`) Bare-Metal 60-Second Benchmark

Measure bare-metal Tensor G5 (`buenos_a0`) compute, memory bandwidth, and `libopenbuenos` mock/silicon throughput with zero proprietary vendor `.so` dependencies.

---

## Prerequisites

- **Host (macOS or Linux)**: `python3` + `numpy`, `cmake`, `clang` (for `libopenbuenos` host mock build).
- **Optional On-Device Silicon**: Unlocked Pixel 10 series (`buenos_a0`, `/dev/edgetpu-soc` `buenos.ko` ABI `1.4`), `adb`, Android NDK r26, and locally extracted `dive_interp_rv64.bin` (never distributed in this repo).

---

## 60-Second Copy-Paste Commands

### 1. Host Mock + `freetpu-bench` Baseline (Zero Device Required)

```bash
git clone https://github.com/saariuslystoned/freetpu.git
cd freetpu

# Run compute + bandwidth + local comms under Tensor G5 buenos_mock profile
python3 bench/freetpu_bench.py \
  --suite all \
  --backend buenos_mock \
  --json runs/buenos-mock.json \
  --markdown runs/buenos-mock.md

cat runs/buenos-mock.md
```

### 2. Verify `libopenbuenos` Host Mock Suite

```bash
cmake -S libopenbuenos -B libopenbuenos/build/cmake-host -DCMAKE_BUILD_TYPE=Release
cmake --build libopenbuenos/build/cmake-host -j
ctest --test-dir libopenbuenos/build/cmake-host --output-on-failure
```

### 3. Live Pixel Tensor G5 Silicon (`rango` / `adb` Root Shell)

```bash
# Cross-build libopenbuenos + standalone bare-metal runner for Android ARM64 (android-34)
# (`verify-baremetal` also audits static ELF DT_NEEDED + undefined dlopen/dlsym symbols)
make -C libopenbuenos verify-header android verify-android
make -C runtime verify-baremetal

# Push runner, locally extracted DIVE interpreter, and a local compiled .dgc0 + input onto your own unlocked device
adb push runtime/build/tpu_baremetal_runner /data/local/tmp/tpu_open_stack/
adb push /path/to/local/dive_interp_rv64.bin /data/local/tmp/tpu_open_stack/dive_interp_rv64.bin
adb push /path/to/local/model.dgc0 /data/local/tmp/tpu_open_stack/model.dgc0
adb push /path/to/local/input.bin /data/local/tmp/tpu_open_stack/input.bin

# Run bare-metal silicon benchmark and capture live /proc/<pid>/maps while the workload executes
adb shell 'su -c "rm -f /data/local/tmp/tpu_open_stack/runner-maps.txt /data/local/tmp/tpu_open_stack/run.json; \
  OPENBUENOS_DIVE_INTERP_PATH=/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin \
  OPENBUENOS_PERF_MODE=5 \
  /data/local/tmp/tpu_open_stack/tpu_baremetal_runner \
    --dgc0 /data/local/tmp/tpu_open_stack/model.dgc0 \
    --in /data/local/tmp/tpu_open_stack/input.bin \
    --iters 500 --json > /data/local/tmp/tpu_open_stack/run.json & \
  pid=\$!; \
  while kill -0 \$pid 2>/dev/null; do \
    if grep -q tpu_baremetal_runner /proc/\$pid/maps 2>/dev/null; then \
      cat /proc/\$pid/maps > /data/local/tmp/tpu_open_stack/runner-maps.tmp 2>/dev/null && \
      mv /data/local/tmp/tpu_open_stack/runner-maps.tmp /data/local/tmp/tpu_open_stack/runner-maps.txt; \
      sleep 0.05; \
    fi; \
  done; \
  wait \$pid"'

# Verify zero proprietary Google/vendor .so libraries loaded at runtime (/proc/<pid>/maps catches transitive + dlopen loads)
# and confirm the runner's in-process /proc/self/maps audit reported maps_has_google_tpu_libs=false
adb shell 'su -c "test -s /data/local/tmp/tpu_open_stack/runner-maps.txt && \
  grep -q tpu_baremetal_runner /data/local/tmp/tpu_open_stack/runner-maps.txt && \
  ! grep -iE \"edgetpu|darwinn|tachyon|libgxp|tflite|tensorflow|/vendor/|google\" /data/local/tmp/tpu_open_stack/runner-maps.txt && \
  grep -q \"\\\"maps_has_google_tpu_libs\\\": false\" /data/local/tmp/tpu_open_stack/run.json"'
```

---

## What Good Looks Like

| Metric | Verified Tensor G5 Silicon (`PM=5` @ `1,286.4 MHz` + FW WakeLock) | Reference Source |
| --- | --- | --- |
| **Peak INT8 Throughput (`64x1536x1536`)** | **`993.4 GOPS`** (`304 us` hw, `5.55 TOPS/W` TPU rail) | [`docs/PHASE6_PERF_REPORT.md`](../docs/PHASE6_PERF_REPORT.md) |
| **Weight DMA Bandwidth (`1x2048x2048`)** | **`27.96 GB/s`** (`150 us` hw / `395 us` wall) | [`docs/PHASE6_PERF_REPORT.md`](../docs/PHASE6_PERF_REPORT.md) |
| **Native `W8A16 (BF16)` (`1x64x32`)** | **`411–426 us`** hw (`32/32` bit-exact `FP32`, `max_abs_err = 0.0`) | [`docs/hardware-errata.md`](../docs/hardware-errata.md) |
| **Zero-DRAM Chained Fusion (`0xa40`)** | **`60.8–72.8 us`** hw (`fuse_fc64_silu_fc64`, `fuse_swiglu_64`, `fuse_qkv_64`) | [`docs/PHASE6_PERF_REPORT.md`](../docs/PHASE6_PERF_REPORT.md) |
| **LLM Inference (`Qwen2.5-1.5B-Q8_0`)** | **`118.06 t/s`** `pp128`, **`5.85 t/s`** `tg64` (`-21.5%` system `mJ/tok` vs CPU on `pp512`) | [`docs/PHASE6_PERF_REPORT.md`](../docs/PHASE6_PERF_REPORT.md) |
| **Runtime `/proc/<pid>/maps` & ELF Audit** | `maps_has_google_tpu_libs = false` (`0` vendor `.so`s mapped in `/proc/<pid>/maps`; `DT_NEEDED` strictly system libs) | [`docs/PHASE6_PERF_REPORT.md`](../docs/PHASE6_PERF_REPORT.md) |

> Note: `--backend buenos_mock` measures your **host CPU** executing FreeTPU's FP32 reference GEMM/SiLU (`gemm_fp32_ref` / `fused_gemm_silu_fp32_ref`) and host INT8 FP32-RNE requantization (`row_type: measured`) alongside the published Tensor G5 silicon baseline (`row_type: published-reference`).

---

## Troubleshooting & Errata

- **Stuck / `-1` Timeout on `INT4` Models**: Tensor G5 (`buenos_a0`) has no native 4-bit nibble-unpack DMA unit (`0x980 w8[21]` is a `1:2` `uint8 -> BF16` expander). Always unpack `Q4_0` once at model load time into cached `uint8` DMA-BUFs (`1 B/weight`). See [`docs/hardware-errata.md`](../docs/hardware-errata.md).
- **Slow `~4–6 ms` Silicon Latency**: Ensure `OPENBUENOS_PERF_MODE=5` (`1,286.4 MHz` core OPP + Litebuf firmware WakeLock); `PM=0` defaults to the `38.4 MHz` sleep clock.
- **Missing `dive_interp_rv64.bin`**: FreeTPU does not ship proprietary firmware. Extract `ext[1]` locally from a `.dgc0` compiled on your own device via `ob_gemm_extract_dive_interp_from_dgc0` (see [`README.md`](../README.md)).
