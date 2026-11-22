# Playbook 03: Dual-Pixel Wired Cluster (`nccl-tests`-Style Hop & Split-Layer RPC)

Benchmark `nccl-tests`-style activation-hop latency (`us`) and throughput (`GB/s`) across `16 KB – 4 MB` PP/MoE payloads, and run split-layer `llama.cpp` (`EDGETPU` + `RPC`) across two wired Pixel 10 devices (`32 GB` pooled RAM).

---

## Prerequisites

- **Loopback / Local Test**: `python3` only (zero extra dependencies).
- **Dual-Node Wired Test**: Two Pixel 10 devices (or host + Pixel) linked via direct **USB-C 3.2 NCM / USB-Ethernet (`2.5GbE/5GbE`)** or `adb forward`/`reverse` TCP bridge.

---

## 60-Second Copy-Paste Commands

### 1. Local `nccl-tests`-Style Loopback Baseline (1 Command)

```bash
python3 bench/freetpu_bench.py \
  --suite comms \
  --json runs/comms-loopback.json \
  --markdown runs/comms-loopback.md

cat runs/comms-loopback.md
```

### 2. Two-Node Wired USB-Ethernet / NCM Activation-Hop Sweep

On **Node 2 (Receiver / Echo Server)** — e.g., Pixel 10 Pro XL (`192.168.77.2`):

```bash
python3 bench/freetpu_bench.py --comms-server --comms-host 192.168.77.2 --comms-port 50055
```
*(Or bind `--comms-host 127.0.0.1` when tunneling over an `adb forward` USB bridge.)*

On **Node 1 (Driver)** — e.g., Pixel 10 Pro Fold or Linux/macOS host:

```bash
python3 bench/freetpu_bench.py \
  --suite comms \
  --comms-host 192.168.77.2 \
  --comms-port 50055 \
  --json runs/comms-wired.json \
  --markdown runs/comms-wired.md
```

*(If testing over an `adb` USB bridge instead of native USB-Ethernet, run `adb -s <node2_serial> forward tcp:50055 tcp:50055` and point `--comms-host 127.0.0.1 --comms-port 50055`.)*

### 3. Split-Layer Dual-Phone `llama.cpp` (`GGML_EDGETPU=ON` + `GGML_RPC=ON`)

```bash
# Cross-build llama-bench & ggml-rpc-server for Android ARM64
./llama-cpp-edgetpu/build.sh android

# Node 2: start clean-room EDGETPU RPC server on the dedicated USB-Ethernet link (zero Google .so)
adb -s <node2_serial> shell "su -c 'OPENBUENOS_PERF_MODE=5 \
  /data/local/tmp/tpu_open_stack/ggml-rpc-server -d EDGETPU -H 192.168.77.2 -p 50052 -t 4'"

# Node 1: split layers across Node 2 RPC + local EDGETPU
adb -s <node1_serial> shell "su -c 'OPENBUENOS_PERF_MODE=5 \
  /data/local/tmp/tpu_open_stack/llama-bench \
  -m /data/local/tmp/models/model.gguf --rpc 192.168.77.2:50052 -ngl 14 -p 128 -n 32 -t 4'"
```

---

## Link Performance Reference

| Transport Path | Observed `16 KB–64 KB` RTT / Hop | Throughput Regime | Notes |
| --- | --- | --- | --- |
| **Loopback TCP (`127.0.0.1`)** | `~20–45 us` 1-way (`~40–90 us` RTT) | `~1–3 GB/s` | Baseline software stack floor |
| **Loopback `ggml-rpc` (`rango`)** | `512–1,774 us` RPC RTT | — | `pp128 = 45.50 t/s` (`Qwen2.5-1.5B-Q8_0`) |
| **USB 2.0 ADB Double-Hop Bridge** | `5,159–10,874 us` RPC RTT | `<40 MB/s` | Bottlenecked by USB 2.0 hub + ADB relay |
| **USB-C 3.2 NCM / 2.5GbE Direct** | `<500 us` target | `~0.3–1.1 GB/s` | Eliminates ADB double-hop for `>16 GB` models |

---

## Clean-Room & Topology Checks

- **Verify zero vendor `.so`s on RPC server**: read `/proc/$pid/maps` inside the device shell without host subshell expansion and verify the read succeeded before asserting zero vendor matches:
  `adb -s <node2_serial> shell 'su -c "pid=\$(pidof -s ggml-rpc-server) && [ -n \"\$pid\" ] && cat /proc/\$pid/maps"' | tee /tmp/rpc-maps.txt >/dev/null && test -s /tmp/rpc-maps.txt && ! grep -E "edgetpu|darwinn|google" /tmp/rpc-maps.txt`
- **Avoid USB 2.0 hubs**: Check link speed (`ioreg` on macOS or `lsusb -t` on Linux) — USB 2.0 (`480 Mbps`) turns weight streaming and PP activation hops into the primary bottleneck.
