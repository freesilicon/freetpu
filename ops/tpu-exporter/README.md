# `tpu-exporter` — Prometheus Telemetry for Cloud TPUs

`tpu_exporter.py` is a zero-dependency Python 3 Prometheus exporter (analogous to NVIDIA `dcgm-exporter`) for VFIO-backed Google Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium). Legacy `/dev/accel*` Cloud TPU hosts (`v2`, `v3`, `v4` using Google's `accel` driver) are recognized in PCI inventory (`freetpu_tpu_info`), while VFIO readiness (`freetpu_tpu_health_status`) and `/dev/vfio/*` PID scanning apply to VFIO-backed hosts.

## What It Discovers & Exports

1. **PCI & VFIO Topology (`/sys/bus/pci/devices/*` + `/dev/vfio/*`)**
   - Filters Google PCI vendor `0x1ae0` and maps known TPU device IDs (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006e`, `0x006f`).
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
- **Workload Sidecar (`ops/tpu-cdi/manifests/k8s-tpu-workload-telemetry-sidecar.yaml`):** Because `libtpu` serves live runtime metrics on `localhost:8431` inside each workload Pod, build the sidecar image with `docker build --build-arg INSTALL_TPU_INFO=1 -t <registry>/freetpu-tpu-exporter:tpu-info ops/tpu-exporter` and co-locate it in the TPU workload Pod. `prometheus-scrape.yml` attaches stable `node` and `collector_role` labels so `promql-queries.md` deduplicates physical chips per `(generation, node, pci_bdf)` when both DaemonSet and sidecars run on the same node.

See [prometheus-scrape.yml](prometheus-scrape.yml) for Prometheus scrape configuration and [promql-queries.md](promql-queries.md) for billing and utilization PromQL queries.
