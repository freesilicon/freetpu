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

- **Node DaemonSet (`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`):** Runs stock `python:3.12-slim` without `tpu-info` outside workload Pod network namespaces; exports **PCI/VFIO inventory and hostPID active-PID telemetry only**. On GKE, host-wide `/proc/<pid>/fd` scanning requires both `CAP_SYS_PTRACE` and `securityContext.appArmorProfile.type: Unconfined` (configured in the DaemonSet manifest) because GKE's default AppArmor profile blocks cross-profile ptrace reads.
- **Workload Sidecar (`ops/tpu-cdi/manifests/k8s-tpu-workload-telemetry-sidecar.yaml`):** Because `libtpu` serves live runtime metrics on `127.0.0.1:8431` inside each workload Pod's network namespace, build the sidecar image with `docker build --build-arg INSTALL_TPU_INFO=1 -t <registry>/freetpu-tpu-exporter:tpu-info ops/tpu-exporter` and run it with `--no-pid-scan --runtime-metrics-addr 127.0.0.1:8431` in the `ml-workloads` namespace (matching the sample DRA manifests and diagnostic scrape allowlist). The sidecar shares only the Pod network namespace (no `shareProcessNamespace`, no `SYS_PTRACE`, and no `/dev/vfio` mount), querying `tpu_info.metrics` directly over loopback with a hard deadline and never falling back to `/proc` PID scans. Billing/node scraping (`prometheus-scrape.yml`) runs ONLY trusted `collector_role="node"` jobs (`freetpu-node-exporter` in `kube-system`), while tenant sidecars are scraped by a **separate diagnostic Prometheus instance** using [prometheus-scrape-diagnostics.yml](prometheus-scrape-diagnostics.yml) (namespace-scoped to `ml-workloads`, `follow_redirects: false`, response/sample/label limits, stripping `pci_bdf` and bounding `chip`/`runtime_chip` to `0-7`). Financial and chargeback queries in `promql-queries.md` consume **exclusively** operator-controlled `collector_role="node"` metrics.

See [prometheus-scrape.yml](prometheus-scrape.yml) for trusted billing/node Prometheus scraping, [prometheus-scrape-diagnostics.yml](prometheus-scrape-diagnostics.yml) for isolated diagnostic sidecar scraping, and [promql-queries.md](promql-queries.md) for billing and diagnostic PromQL queries.
