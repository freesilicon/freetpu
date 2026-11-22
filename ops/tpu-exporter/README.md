# `tpu-exporter` — Prometheus Telemetry for Cloud TPUs

`tpu_exporter.py` is a zero-dependency Python 3 Prometheus exporter (analogous to NVIDIA `dcgm-exporter`) for VFIO-backed Google Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium). Legacy `/dev/accel*` Cloud TPU hosts (`v2`, `v3`, `v4` using Google's `accel` driver) are recognized in PCI inventory (`freetpu_tpu_info`), while VFIO readiness (`freetpu_tpu_health_status`) and `/dev/vfio/*` PID scanning apply to VFIO-backed hosts.

## What It Discovers & Exports

1. **PCI & VFIO Topology (`/sys/bus/pci/devices/*` + `/dev/vfio/*`)**
   - Filters Google PCI vendor `0x1ae0` and maps known TPU device IDs (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006e`, `0x006f`). Note: upstream `tpu-info` omits v6e PF `0x006e` (recognizing v6e VF `0x006f`), so PF-only `0x006e` hosts export sysfs/VFIO inventory and active PIDs without `tpu-info` HBM/duty-cycle metrics.
   - Resolves IOMMU group, bound kernel driver (`vfio-pci`), `/dev/vfio/<group>` node, and `/dev/vfio/vfio` container readiness on VFIO-backed hosts.
2. **Runtime Telemetry (`tpu-info` + `/proc/*/fd` fallback)**
   - Parses both modern `tpu-info` dashboard (`TPU Device Status`, `TPU Process Monitor`) and classic (`TPU Chips`, `TPU Runtime Utilization`) tables for HBM used/total bytes, MXU/TensorCore duty cycle `%`, and active PIDs.
   - Leaves utilization metrics unset when `tpu-info` reports `N/A` or unrecognized output (never fabricates zero utilization).
   - Falls back cleanly to sysfs + `/proc/<pid>/fd` VFIO handle scanning when `tpu-info` is unavailable.

## Usage

```bash
# HTTP /metrics server on localhost:9400 (default; pass --host 0.0.0.0 in containers
# and --health-port 9401 to keep liveness/readiness probes independent of scrape saturation)
python3 ops/tpu-exporter/tpu_exporter.py --port 9400 --health-port 9401

# One-shot Prometheus text exposition snapshot (CI / ephemeral Colab runs)
python3 ops/tpu-exporter/tpu_exporter.py --once

# Structured JSON snapshot
python3 ops/tpu-exporter/tpu_exporter.py --once --json
```

### Live Telemetry in Kubernetes vs. Node DaemonSet

- **Node DaemonSet (`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`):** Runs stock `python:3.12-slim` without `tpu-info` outside workload Pod network namespaces and passes `--no-pid-scan --tpu-info-bin ""` by default (no `hostPID`, no `CAP_SYS_PTRACE`, no `Unconfined` AppArmor), exporting **PCI/VFIO hardware inventory (including `vfio_group_exclusive`) and VFIO readiness only**. Because GKE Standard accepts `NetworkPolicy` resources even when enforcement is disabled—and Kubernetes `NetworkPolicy` permits traffic originating from the exporter's own node (including same-node `hostNetwork: true` pods on CNIs that treat them as node traffic)—remote host PID telemetry (`freetpu_tpu_active_pids`) is an explicit opt-in: enable `hostPID: true`, `CAP_SYS_PTRACE`, `securityContext.appArmorProfile.type: Unconfined` (on GKE), and remove `--no-pid-scan` only on clusters with verified NetworkPolicy enforcement paired with admission controls forbidding tenant `hostNetwork: true` pods, or by binding `--host 127.0.0.1` behind an authenticating proxy such as `kube-rbac-proxy`. The manifest includes a companion `NetworkPolicy` (`freetpu-node-agent-metrics`) restricting ingress on TCP `:9400` to the trusted `monitoring` Prometheus scraper.
- **Live Runtime Telemetry (`tpu-info` / `--no-pid-scan`):** On standalone Cloud TPU VMs (or an operator-managed container sharing a workload's loopback `127.0.0.1:8431` `libtpu` endpoint), build the image with `docker build --build-arg INSTALL_TPU_INFO=1 -t <registry>/freetpu-tpu-exporter:tpu-info ops/tpu-exporter`. The image's `ENTRYPOINT` sets `--host 0.0.0.0 --port 9400 --health-host 127.0.0.1 --health-port 9401 --no-pid-scan` so containers in Docker's default private PID namespace never misreport host PID availability (`freetpu_tpu_pid_telemetry_available=1` with zero PIDs); passing `--runtime-metrics-addr 127.0.0.1:8431` preserves those defaults, while host PID collection in the image requires `--pid=host --cap-add=SYS_PTRACE` plus an explicit `--pid-scan` argument. When invoking `tpu_exporter.py` directly outside the container image, pass the full flag list (`--host 0.0.0.0 --port 9400 --health-host 127.0.0.1 --health-port 9401 --no-pid-scan --runtime-metrics-addr 127.0.0.1:8431`).

See [prometheus-scrape.yml](prometheus-scrape.yml) for trusted node Prometheus scraping and [promql-queries.md](promql-queries.md) for fleet inventory, utilization, and observational PromQL queries.
