# Track A: Dual-Pixel 10 Wired TPU Cluster Playbook & Proof (`2× Tensor G5 buenos_a0`)

**Target Hardware**: Google Pixel 10 Pro Fold (`rango`) + Google Pixel 10 Pro XL (`mustang`), both running Android 17 (`CP3A.261005.005`, `buenos_a0`, `15,575 MiB` physical RAM each → **`30.4 GiB` combined RAM**).  
**Interconnect**: Direct single-cable USB-C 3.2 SuperSpeed (`5 Gbps`) CDC-NCM Ethernet (`10.99.0.1/30 ↔ 10.99.0.2/30`, zero Wi-Fi inference traffic).  
**Clean-Room Invariant**: Zero Google/vendor `.so` dependencies (`DT_NEEDED = [libdl.so, libm.so, libc.so]` on all 5 stripped Android ARM64 binaries; zero vendor libraries mapped in `/proc/<pid>/maps` on both nodes).

---

## 1. Software Architecture (`GGML_EDGETPU=ON` + `GGML_RPC=ON`)

Upstream `ggml-org/llama.cpp` (`b12608`) treats `EDGETPU` as a host-accessible `ACCEL` backend (`supports_op == true` for `MUL_MAT` and post-`MUL_MAT` 1D bias `ADD`, routing glue ops like `RMS_NORM`, `ROPE`, `SOFT_MAX`, `SILU`, `GET_ROWS` to ARM64 CPU). Three targeted changes enable multi-node `EDGETPU` pipeline parallelism:

1. **Android ARM64 `EDGETPU` + `RPC` Cross-Build (`llama-cpp-edgetpu/build.sh`)**:
   - Enables `-DGGML_EDGETPU=ON -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF` and builds/strips/audits `ggml-rpc-server` alongside `llama-cli`, `llama-bench`, `llama-completion`, and `test-backend-ops`.
2. **Unified-RAM Device Memory Reporting (`llama-cpp-edgetpu/ggml-edgetpu.cpp`)**:
   - `ggml_backend_edgetpu_device_get_memory` reports real Linux/Android physical memory (`sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)`) and `/proc/meminfo` `MemAvailable`, so `ggml-rpc-server -d EDGETPU` advertises accurate remote RAM capacity (`15,575 MiB` total, `~10,400 MiB` free) to the coordinator.
3. **Server-Side `ACCEL` + `CPU` Hybrid Subgraph Execution (`llama-cpp-edgetpu/patches/llama-cpp-b12608-rpc-accel-fallback.patch`)**:
   - When `ggml-rpc-server` serves a host-accessible `ACCEL` device (`-d EDGETPU`), `rpc_server::compute_graph_with_accel_fallback` partitions each offloaded subgraph via `ggml_graph_view` into contiguous `EDGETPU`-supported runs (`MUL_MAT`, `MUL_MAT_ID`, and bias `ADD`, preserving sibling `Q/K/V` and `gate/up` batching) and validated server-local `CPU` fallback runs (`RMS_NORM`, `SCALE`, `ADD`/`SUB`/`MUL`/`DIV`, `MUL_MAT`, and view/no-op ops). Per-device `hello` flags advertise the hybrid capability set in `ggml_backend_rpc_device_supports_op` so remaining ops (`ROPE`, `SOFT_MAX`, `GLU`/`SILU`, KV-cache ops) run on the coordinator CPU while non-ACCEL RPC devices retain full upstream op support.

---

## 2. Reproducible Wired Cluster Playbook

### 2.1 Build & Audit Android ARM64 Binaries

```bash
make -C libopenbuenos host android verify
cd llama-cpp-edgetpu
./build.sh host
./build.sh android
```

Verified `llvm-readelf -d` `DT_NEEDED` across all stripped ARM64 binaries (`ggml-rpc-server` `1.7M`, `llama-bench` `6.1M`, `llama-completion` `8.3M`, `llama-cli` `16M`, `test-backend-ops` `3.2M`):
```text
Shared library: [libdl.so]
Shared library: [libm.so]
Shared library: [libc.so]
```
Host mock verification (`OPENBUENOS_MOCK=1`): `test-backend-ops -b EDGETPU -o MUL_MAT` → **`327/327` PASS**.

### 2.2 Bring Up Direct 5 Gbps USB-C CDC-NCM Link (Zero Wi-Fi Data)

Connect both rooted Pixel 10s directly with a USB-C 3.2 / Thunderbolt 5 cable (if using Wireless Debugging during initial link setup, disable Wi-Fi on Node 2 before starting the unauthenticated RPC server so only the point-to-point USB-C NCM link is active). On the USB `device` side (`mustang`), enable CDC-NCM and assign a point-to-point `/30` subnet with high-priority policy routing (`table 200`):

```bash
# Node 2 (mustang — USB device side, super-speed):
su -c '
  svc usb setFunctions ncm
  sleep 2
  ip addr flush dev ncm0 2>/dev/null || true
  ip link set ncm0 up
  ip addr add 10.99.0.2/30 dev ncm0
  ip route replace 10.99.0.0/30 dev ncm0 table 200
  ip rule del to 10.99.0.0/30 lookup 200 2>/dev/null || true
  ip rule add pref 100 to 10.99.0.0/30 lookup 200
'

# Node 1 (rango — USB host side, 5000M SuperSpeed):
su -c '
  ip link set usb0 up
  ip addr add 10.99.0.1/30 dev usb0 2>/dev/null || true
  ip route replace 10.99.0.0/30 dev usb0 src 10.99.0.1 table 200
  ip rule del to 10.99.0.0/30 lookup 200 2>/dev/null || true
  ip rule add pref 100 to 10.99.0.0/30 lookup 200
  echo on > /sys/bus/usb/devices/2-1/power/control 2>/dev/null || true
  echo 50 > /sys/class/net/usb0/cdc_ncm/tx_timer_usecs
  ping -c 10 -i 0.02 10.99.0.2
'
```

Measured direct link: **`1.49 ms` min / `2.35 ms` avg** ping RTT (`0%` loss), **`1.01–3.77 ms`** sync RPC RTT (`GET_DEVICE_MEMORY` / `ALLOC_BUFFER`), and **`469–810 µs`** async `GRAPH_COMPUTE` push.

### 2.3 Run `>16 GB` Model Across `2× Tensor G5` (`Qwen2.5-32B-Instruct-Q4_0`, `17.43 GiB`)

Bound the per-node `ext[2]` DMA-BUF LRU weight cache (`OPENBUENOS_WEIGHT_CACHE_MB=128`, `OPENBUENOS_POOL_MB=64`) to stay comfortably below Android LMKD limits while holding `~9 GiB` of model weights per phone:

```bash
# Node 2 (mustang — disable Wi-Fi BEFORE launching the unauthenticated RPC server so only ncm0 is active):
su -c '
  svc wifi disable
  OPENBUENOS_VERBOSE=1 OPENBUENOS_WEIGHT_CACHE_MB=128 OPENBUENOS_WEIGHT_CACHE_ENTRIES=256 OPENBUENOS_POOL_MB=64 \
  /data/local/tmp/llama-edgetpu-rpc/ggml-rpc-server -d EDGETPU -H 10.99.0.2 -p 50052 -t 4
'

# Node 1 (rango — split 34/65 layers onto mustang EDGETPU, keep 31/65 layers on rango EDGETPU):
# Note: `--no-kv-offload` keeps KV-cache `SET_ROWS` writes on the coordinator while offloading
# model weights and `MUL_MAT` projections to Node 2's EDGETPU RPC server.
su -c '
  OPENBUENOS_VERBOSE=1 OPENBUENOS_WEIGHT_CACHE_MB=128 OPENBUENOS_WEIGHT_CACHE_ENTRIES=256 OPENBUENOS_POOL_MB=64 \
  /data/local/tmp/llama-edgetpu-rpc/llama-completion \
    -m /data/local/tmp/llama-edgetpu/models/Qwen2.5-32B-Instruct-Q4_0.gguf \
    --rpc 10.99.0.2:50052 -ngl 34 --no-kv-offload -c 256 -b 64 -ub 64 -t 4 -no-cnv \
    -p "The capital of France is" -n 8 --temp 0 --no-display-prompt
'
```

---

## 3. Verified Live Silicon Results

### 3.1 `Qwen2.5-32B-Instruct-Q4_0` (`17.43 GiB` / `18.71 GB`, Exceeds Single 16 GB Phone)

- **Artifact**: `Qwen2.5-32B-Instruct-Q4_0.gguf` (`18,711,010,176 B`, `SHA-256: edc1aa9db2d886329413ca3db40852376736a220c660e7a1b2a166aacf9d9094`).
- **Greedy Output (`temp=0`, `CLI_RC=0`)**: `" Paris. Paris is located in the northern"`
- **Per-Node Pooled Memory Split (`34/65` layers offloaded to Node 2 `mustang`, `31/65` local on Node 1 `rango`)**:

| Node | Role / Backend | Model Buffer | Compute Buffer | Peak `VmHWM` | `VmSwap` |
| --- | --- | ---: | ---: | ---: | ---: |
| **Node 1 (`rango`)** | Local `EDGETPU` + `CPU` | `7,500.98 MiB` (`EDGETPU`) + `1,094.50 MiB` (`CPU_Mapped`) | `14.75 MiB` + `13.03 MiB` | **`10,309,632 kB` (`9.83 GiB`)** | `38,676 kB` (`37.8 MiB`) |
| **Node 2 (`mustang`)** | Remote `RPC0` → `EDGETPU` (`10.99.0.2:50052`) | **`9,242.86 MiB` (`9.03 GiB`)** | `38.38 MiB` | **`9,550,644 kB` (`9.11 GiB`)** | **`0 kB`** |
| **Combined Cluster** | **`2× Tensor G5 (buenos_a0)`** | **`17,838.34 MiB` (`17.42 GiB`)** | `66.16 MiB` | **`19,860,276 kB` (`18.94 GiB` / `20.34 GB`)** | `37.8 MiB` |

- **Silicon Telemetry (`PM=5 @ 1,286.4 MHz`, `mock=0`, `cpu_fallback_slices=0`)**:
  - **Node 1 (`rango` `EDGETPU`)**: `calls=1881`, `slices=1881`, `emitter_slices=1881`, `cpu_fallback_slices=0`, `batched_groups=558`, `batched_ops=1116`, `bias_adds=1215`, `tiles=45817`, `tile_hits=45642` (`99.62%` graph cache hit), **`hw_us = 8,174,892` (`8.17 s` across 12 tokens ≈ `0.68 s/tok` pure TPU silicon per node)**, `wall_us = 479.75 s`.
  - **Node 2 (`mustang` `EDGETPU`)**: live `Q4_0` silicon dispatches (`5120×5120`: `hw_us = 3,763–3,786 µs`; `27648×5120`: `hw_us = 19,702–19,716 µs`).
  - **End-to-End Timing**: `load = 223.45 s` (`9.03 GiB` streamed over 5 Gbps USB-C NCM), `prompt eval = 0.06 t/s` (`5 tok`), `eval = 0.01 t/s` (`7 runs`).
  - **Bottleneck Diagnosis**: With `OPENBUENOS_WEIGHT_CACHE_MB=128` protecting 16 GB LMKD, every token evicted and re-packed `Q4_0 → ext[2]` weights on CPU (`wall_us / hw_us = 58.7×`). Pure Tensor G5 silicon ran ~50× faster (`~0.68 s/tok` per node), directly motivating **single-copy pre-packed `ext[2]` residency** for fast 32B decode.

### 3.2 `Qwen2.5-1.5B-Q8_0` (`1.76 GiB`) Benchmark Matrix (`llama-bench -p 128 -n 32 -r 2 -t 4`, plus `--no-kv-offload 1` for `RPC → EDGETPU` rows)

| Topology | Interconnect | `ngl` | Model Load (`853 MiB`) | Sync Wire RTT | `pp128` (`t/s`) | `tg32` (`t/s`) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `rango` Local `EDGETPU` | Single phone | `0` | local | — | `109.79 ± 0.79` | `5.52 ± 0.82` |
| `rango` Loopback `RPC → EDGETPU` | `127.0.0.1:50052` | `99` | local TCP | `0.51–1.77 ms` | `45.50 ± 0.21` | `1.24 ± 0.19` |
| `rango` `EDGETPU` + `mustang` `CPU` (unrooted) | USB 2.0 ADB tunnel | `14` | `30.95 s` | `5.16–10.87 ms` | `63.49 ± 0.28` | `2.26 ± 0.19` |
| **`rango` `EDGETPU` + `mustang` `EDGETPU` (`2× buenos_a0`)** | **Direct 5 Gbps USB-C NCM (`10.99.0.2`)** | **`14`** | **`8.54 s` (`4.15×` faster)** | **`1.01–3.77 ms`** (`469–810 µs` async push) | **`45.05 ± 0.67`** | **`1.03 ± 0.11`** |

### 3.3 Live `/proc/<pid>/maps` Zero-Vendor-`.so` Audit (Both Nodes)

During the `32B` run, every mapped `.so` in both `llama-completion` (Node 1 `rango`) and `ggml-rpc-server` (Node 2 `mustang`) was strictly AOSP Bionic / system C++ runtime — zero `libedgetpu*`, `libdarwinn*`, `libgxp*`, `liblitert*`, or `/vendor/*` entries:

```text
/apex/com.android.runtime/lib64/bionic/libc.so
/apex/com.android.runtime/lib64/bionic/libdl.so
/apex/com.android.runtime/lib64/bionic/libm.so
/system/lib64/libc++.so
/system/lib64/libnetd_client.so
```
