# `tpu-exporter` — Prometheus Telemetry for Cloud TPUs

`tpu_exporter.py` is a zero-dependency Python 3 Prometheus exporter (analogous to NVIDIA `dcgm-exporter`) for VFIO-backed Google Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium). Legacy `/dev/accel*` Cloud TPU hosts (`v2`, `v3`, `v4` using Google's `accel` driver) are recognized in PCI inventory (`freetpu_tpu_info`), while VFIO readiness (`freetpu_tpu_health_status`) and `/dev/vfio/*` PID scanning apply to VFIO-backed hosts.

## What It Discovers & Exports

1. **PCI & VFIO Topology (`/sys/bus/pci/devices/*` + `/dev/vfio/*`)**
   - Filters Google PCI vendor `0x1ae0` and maps known TPU device IDs (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006e`, `0x006f`, and `0x0075` for `tpu7x` pending live hardware confirmation `#21`). Note: upstream `tpu-info` CLI discovery omits v6e PF `0x006e` (recognizing v6e VF `0x006f`), while `tpu_exporter.py` maps `0x006e` directly to `TpuChip.V6E` and queries `127.0.0.1:8431` via direct gRPC or `tpu_info.metrics` fallback.
   - Resolves IOMMU group, bound kernel driver (`vfio-pci`), `/dev/vfio/<group>` node, `/dev/vfio/vfio` container readiness, and pure `libtpu`-free sysfs PCIe AER error counters (`aer_dev_correctable` -> `freetpu_pcie_aer_correctable_total`, `aer_dev_fatal` -> `freetpu_pcie_aer_fatal_total`).
2. **Runtime Telemetry (Direct gRPC + SeqLock MMF + `tpu-info` / `/proc/*/fd` fallback)**
   - **Direct `libtpu` gRPC (`RuntimeMetricService` on `127.0.0.1:8431`):** Queries `GetRuntimeMetric` in-process over a reused channel when `grpcio` and `tpu_metric_service` proto stubs are installed (eliminating `tpu-info` subprocess churn) for HBM usage/total, MXU duty cycle `%`, and HBM bandwidth utilization `%`.
   - **Lockless SeqLock MMF (`/dev/shm/freetpu_metrics_<bdf>`):** Reads bare-metal `libopentpu.so` counters (`freetpu_shm_metrics_v1` in [freetpu_metrics_shm.h](freetpu_metrics_shm.h)) via read-only `mmap` with torn-read retry (`freetpu_tpu_shm_hbm_allocated_bytes`, `freetpu_tpu_mxu_active_cycles_total`, `freetpu_tpu_dma_bytes_transferred_total`).
   - **GCP Spot preemption metadata (opt-in `--gcp-metadata-preemption`):** Fast-timeout poll of `metadata.google.internal` emitting `gcp_metadata_preempted_status` (`1` = preempted, `0` = active).
   - **Legacy fallback:** Falls back cleanly to `tpu_info.metrics`, `tpu-info` CLI, and `/proc/<pid>/fd` VFIO handle scanning when direct gRPC stubs are absent; leaves utilization unset on `N/A` or unparsed output.

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

- **Node DaemonSet (`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`):** Runs stock `python:3.12-slim` without `tpu-info` outside workload Pod network namespaces and passes `--no-pid-scan --tpu-info-bin ""` by default (no `hostPID`, no `CAP_SYS_PTRACE`, no `Unconfined` AppArmor), exporting **PCI/VFIO hardware inventory (including `vfio_group_exclusive`) and VFIO readiness only**. Because GKE Standard accepts `NetworkPolicy` resources even when enforcement is disabled—and Kubernetes `NetworkPolicy` permits traffic originating from the exporter's own node (including same-node `hostNetwork: true` pods on CNIs that treat them as node traffic)—shared/multi-tenant clusters (including default inventory-only mode) require verified NetworkPolicy enforcement paired with admission controls forbidding tenant `hostNetwork: true` pods, or binding `--host 127.0.0.1` behind an authenticating proxy such as `kube-rbac-proxy`, so tenant workloads cannot starve the exporter's bounded `:9400` scrape worker pool. Remote host PID telemetry (`freetpu_tpu_active_pids`) is an additional explicit opt-in: enable `hostPID: true`, `CAP_SYS_PTRACE`, `securityContext.appArmorProfile.type: Unconfined` (on GKE), and remove `--no-pid-scan` only after verifying that ingress isolation. The manifest includes a companion `NetworkPolicy` (`freetpu-node-agent-metrics`) restricting ingress on TCP `:9400` to the trusted `monitoring` Prometheus scraper.
- **Live Runtime Telemetry (`tpu-info` / `--no-pid-scan`):** On standalone Cloud TPU VMs (or an operator-managed container sharing a workload's loopback `127.0.0.1:8431` `libtpu` endpoint), build the image with `docker build --build-arg INSTALL_TPU_INFO=1 -t <registry>/freetpu-tpu-exporter:tpu-info ops/tpu-exporter`. The image's `ENTRYPOINT` sets `--host 0.0.0.0 --port 9400 --health-host 127.0.0.1 --health-port 9401 --no-pid-scan` so containers in Docker's default private PID namespace never misreport host PID availability (`freetpu_tpu_pid_telemetry_available=1` with zero PIDs); passing `--runtime-metrics-addr 127.0.0.1:8431` preserves those defaults. For host VFIO readiness and opt-in host PID collection in Docker, bind-mount the host sysfs subpaths and `/dev/vfio` read-only (`-v /sys/bus/pci:/sys/bus/pci:ro -v /sys/devices:/sys/devices:ro -v /sys/kernel/iommu_groups:/sys/kernel/iommu_groups:ro -v /dev/vfio:/dev/vfio:ro`), add `--pid=host --cap-add=SYS_PTRACE` (with an AppArmor profile permitting cross-profile `/proc/<pid>/fd` reads), and pass `--pid-scan`. When invoking `tpu_exporter.py` directly outside the container image, pass the full flag list (`--host 0.0.0.0 --port 9400 --health-host 127.0.0.1 --health-port 9401 --no-pid-scan --runtime-metrics-addr 127.0.0.1:8431`).

See [prometheus-scrape.yml](prometheus-scrape.yml) for trusted node Prometheus scraping, [prometheus-alerts.yml](prometheus-alerts.yml) for production alert rules (`TpuHbmOOMWarning`, `TpuMxuDutyCycleCollapse`, `TpuPcieAerFault`, `GcpSpotPreemption`), and [promql-queries.md](promql-queries.md) for fleet inventory, utilization, and observational PromQL queries.
