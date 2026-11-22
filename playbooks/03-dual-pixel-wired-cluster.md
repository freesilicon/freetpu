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

*(If testing over an `adb` USB bridge instead of native USB-Ethernet, run `adb -s <node2_serial> forward tcp:localhost:50055 tcp:50055` for a desktop driver—and when Node 1 is a second Pixel, also bind a loopback-only reverse tunnel `adb -s <node1_serial> reverse tcp:localhost:50055 tcp:50055` and drop non-loopback ingress on TCP `50055`—then point the driver at `--comms-host 127.0.0.1 --comms-port 50055`.)*

### 3. Split-Layer Dual-Phone `llama.cpp` (`GGML_EDGETPU=ON` + `GGML_RPC=ON`)

> **Security Prerequisites (`ggml-rpc-server`)**: Upstream `llama.cpp` `ggml-rpc-server` has **no client authentication or transport encryption** and runs as root here to access `/dev/edgetpu-soc`. Even when bound to `127.0.0.1` over ADB forwarding/reverse, loopback listeners are reachable by other local apps on both Node 2 and Node 1 (and by local processes on the host). **Run this recipe only on dedicated, single-purpose test devices with only trusted apps and a trusted host**, and optionally enforce UID filtering on `lo` (shown below) before launching the server. Only bind directly to a USB-Ethernet interface (e.g. `-H 192.168.77.2`) on an **isolated, air-gapped lab test device** with `iptables` rules restricting TCP `50052` exclusively to the intended peer IP (`192.168.77.1`).

**Prerequisites (Build & Stage Static Binaries + Local Artifacts on Both Phones):**

Requires upstream `ggml-org/llama.cpp` (`b12608`, see [`llama-cpp-edgetpu/README.md`](../llama-cpp-edgetpu/README.md)), Android NDK r26, your locally extracted `dive_interp_rv64.bin`, and a quantized `.gguf` model on your host.

```bash
# 1. Cross-build static Android ARM64 base tree, then reconfigure existing build dir with GGML_RPC=ON
./llama-cpp-edgetpu/build.sh android
cmake llama-cpp-edgetpu/build-android-arm64 -DGGML_RPC=ON
cmake --build llama-cpp-edgetpu/build-android-arm64 --target ggml-rpc-server llama-bench -j

# 2. Stage static ggml-rpc-server + DIVE interpreter on Node 2 (zero vendor .so; libopenbuenos is statically linked)
adb -s <node2_serial> shell "mkdir -p /data/local/tmp/tpu_open_stack"
adb -s <node2_serial> push llama-cpp-edgetpu/build-android-arm64/bin/ggml-rpc-server /data/local/tmp/tpu_open_stack/ggml-rpc-server
adb -s <node2_serial> push /path/to/local/dive_interp_rv64.bin /data/local/tmp/tpu_open_stack/dive_interp_rv64.bin
adb -s <node2_serial> shell "chmod +x /data/local/tmp/tpu_open_stack/ggml-rpc-server"

# 3. Stage static llama-bench + DIVE interpreter + your local .gguf model on Node 1
adb -s <node1_serial> shell "mkdir -p /data/local/tmp/tpu_open_stack /data/local/tmp/models"
adb -s <node1_serial> push llama-cpp-edgetpu/build-android-arm64/bin/llama-bench /data/local/tmp/tpu_open_stack/llama-bench
adb -s <node1_serial> push /path/to/local/dive_interp_rv64.bin /data/local/tmp/tpu_open_stack/dive_interp_rv64.bin
adb -s <node1_serial> push /path/to/local/model.gguf /data/local/tmp/models/model.gguf
adb -s <node1_serial> shell "chmod +x /data/local/tmp/tpu_open_stack/llama-bench"
```

**Terminal 1 — Start Persistent RPC Server on Node 2 (leave running):**

```bash
# Optional hardening on Node 2: allow only root (UID 0) and adbd/shell (UID 2000) on loopback TCP 50052
# (rejects untrusted Android app UIDs >= 10000 before starting the server)
adb -s <node2_serial> shell "su -c 'iptables -N FREETPU_RPC_OUT 2>/dev/null || iptables -F FREETPU_RPC_OUT; \
  iptables -A FREETPU_RPC_OUT -m owner --uid-owner 0 -j RETURN; \
  iptables -A FREETPU_RPC_OUT -m owner --uid-owner 2000 -j RETURN; \
  iptables -A FREETPU_RPC_OUT -j REJECT; \
  iptables -C OUTPUT -o lo -p tcp --dport 50052 -j FREETPU_RPC_OUT 2>/dev/null || \
  iptables -I OUTPUT -o lo -p tcp --dport 50052 -j FREETPU_RPC_OUT'"

# Node 2: bind clean-room EDGETPU RPC server to loopback ONLY (zero Google .so); leave running in Terminal 1
adb -s <node2_serial> shell "su -c 'OPENBUENOS_DIVE_INTERP_PATH=/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin \
  OPENBUENOS_PERF_MODE=5 \
  /data/local/tmp/tpu_open_stack/ggml-rpc-server -d EDGETPU -H 127.0.0.1 -p 50052 -t 4'"
```

**Terminal 2 — Create Loopback-Only ADB Tunnels, Verify Listener, & Run Split-Layer Benchmark on Node 1:**

```bash
# Hardening on Node 1: drop any non-loopback IPv4/IPv6 ingress to TCP 50052 AND allow only root (UID 0)
# and adbd/shell (UID 2000) to connect over loopback before opening the reverse tunnel
adb -s <node1_serial> shell "su -c 'iptables -C INPUT ! -i lo -p tcp --dport 50052 -j DROP 2>/dev/null || \
  iptables -I INPUT ! -i lo -p tcp --dport 50052 -j DROP; \
  ip6tables -C INPUT ! -i lo -p tcp --dport 50052 -j DROP 2>/dev/null || \
  ip6tables -I INPUT ! -i lo -p tcp --dport 50052 -j DROP; \
  iptables -N FREETPU_RPC_OUT 2>/dev/null || iptables -F FREETPU_RPC_OUT; \
  iptables -A FREETPU_RPC_OUT -m owner --uid-owner 0 -j RETURN; \
  iptables -A FREETPU_RPC_OUT -m owner --uid-owner 2000 -j RETURN; \
  iptables -A FREETPU_RPC_OUT -j REJECT; \
  iptables -C OUTPUT -o lo -p tcp --dport 50052 -j FREETPU_RPC_OUT 2>/dev/null || \
  iptables -I OUTPUT -o lo -p tcp --dport 50052 -j FREETPU_RPC_OUT'"

# Tunnel Node 2 loopback -> host loopback -> Node 1 loopback explicitly
# (`tcp:localhost:50052` enforces loopback on both ends, even if host `adb` was started with `-a`, and avoids wildcard `adbd` reverse binds)
adb -s <node2_serial> forward tcp:localhost:50052 tcp:50052
adb -s <node1_serial> reverse tcp:localhost:50052 tcp:50052

# Verify Node 1 listener on TCP 50052 (0xC384) via kernel procfs (works on production Android builds without `ss`).
# Require >=1 loopback LISTEN (0A) socket and zero non-loopback/wildcard listeners; chain with `&&` so `llama-bench`
# does NOT run if listener verification fails.
adb -s <node1_serial> shell "su -c 'listen=\$(grep -ihE \"^[[:space:]]*[0-9]+:[[:space:]]+[0-9A-F]+:C384[[:space:]]+[0-9A-F]+:0000[[:space:]]+0A[[:space:]]\" /proc/net/tcp /proc/net/tcp6 2>/dev/null); \
  printf \"%s\n\" \"\$listen\"; \
  printf \"%s\n\" \"\$listen\" | grep -qiE \"^[[:space:]]*[0-9]+:[[:space:]]+(0100007F|00000000000000000000000001000000|0000000000000000FFFF00000100007F):C384[[:space:]]\" && \
  ! printf \"%s\n\" \"\$listen\" | grep -qivE \"^[[:space:]]*[0-9]+:[[:space:]]+(0100007F|00000000000000000000000001000000|0000000000000000FFFF00000100007F):C384[[:space:]]\"'" && \
adb -s <node1_serial> shell "su -c 'OPENBUENOS_DIVE_INTERP_PATH=/data/local/tmp/tpu_open_stack/dive_interp_rv64.bin \
  OPENBUENOS_PERF_MODE=5 \
  /data/local/tmp/tpu_open_stack/llama-bench \
  -m /data/local/tmp/models/model.gguf --rpc 127.0.0.1:50052 -ngl 14 -p 128 -n 32 -t 4'"
```

---

## Link Performance Reference

| Transport Path | Observed `16 KB–64 KB` RTT / Hop | Throughput Regime | Notes |
| --- | --- | --- | --- |
| **Loopback TCP (`127.0.0.1`)** | `~20–45 us` half-RTT est. (`~40–90 us` RTT) | `~1–3 GB/s` | Baseline software stack floor (ping-pong `RTT / 2` estimate) |
| **Loopback `ggml-rpc` (`rango`)** | `512–1,774 us` RPC RTT | — | `pp128 = 45.50 t/s` (`Qwen2.5-1.5B-Q8_0`) |
| **USB 2.0 ADB Double-Hop Bridge** | `5,159–10,874 us` RPC RTT | `<40 MB/s` | Bottlenecked by USB 2.0 hub + ADB relay |
| **USB-C 2.5GbE Direct Ethernet** | `<500 us` target | `~0.25–0.30 GB/s` | `0.3125 GB/s` raw line-rate ceiling (half-duplex ping-pong) |
| **USB-C 5GbE Direct Ethernet** | `<500 us` target | `~0.50–0.60 GB/s` | `0.625 GB/s` raw line-rate ceiling (half-duplex ping-pong) |
| **USB-C 3.2 NCM Direct (Gadget)** | `<500 us` target | `~0.4–0.5 GB/s` (Gen 1 `5 Gbps`) / `~0.8–1.1 GB/s` (Gen 2 `10 Gbps`) | Depends on negotiated USB 3.2 link speed; eliminates ADB double-hop |

---

## Clean-Room & Topology Checks

- **Verify zero vendor `.so`s on RPC server**: read `/proc/$pid/maps` inside the device shell without host subshell expansion or predictable `/tmp` files, and verify the read succeeded before asserting zero vendor matches:
  `maps=$(adb -s <node2_serial> shell 'su -c "pid=\$(pidof -s ggml-rpc-server) && [ -n \"\$pid\" ] && cat /proc/\$pid/maps"') && [ -n "$maps" ] && ! printf '%s\n' "$maps" | grep -iE "edgetpu|darwinn|tachyon|libgxp|tflite|tensorflow|/vendor/|google"`
- **Avoid USB 2.0 hubs**: Check link speed (`ioreg` on macOS or `lsusb -t` on Linux) — USB 2.0 (`480 Mbps`) turns weight streaming and PP activation hops into the primary bottleneck.
