# Cloud TPU Ops Plumbing: Container CDI & Prometheus Telemetry

FreeTPU Pillar 6 brings GPU-grade operational ergonomics (`dcgm-exporter` + `nvidia-container-toolkit` + Kubernetes device plumbing) to Google Cloud TPU VMs (`v4`, `v5e`, `v5p`, `v6e` Trillium) using zero-dependency Python tooling over standard Linux sysfs (`vendor=0x1ae0`) and VFIO (`/dev/vfio/*`).

---

## Architecture Overview

| NVIDIA GPU Stack | FreeTPU Cloud TPU Equivalent | Path |
| --- | --- | --- |
| `nvidia-dcgm-exporter` | `tpu-exporter` (`:9400/metrics`) | [`ops/tpu-exporter/tpu_exporter.py`](../../ops/tpu-exporter/tpu_exporter.py) |
| `nvidia-ctk cdi generate` | `freetpu-cdi generate` (CNCF CDI `v0.6.0`) | [`ops/tpu-cdi/freetpu_cdi.py`](../../ops/tpu-cdi/freetpu_cdi.py) |
| `k8s-device-plugin` / GPU Operator | K8s DaemonSet + DRA / CDI Pod Manifests | [`ops/tpu-cdi/manifests/`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) |

Cloud TPU VMs expose each TPU chip as a PCI function (`vendor=0x1ae0`) bound to the standard Linux `vfio-pci` kernel driver, with per-IOMMU-group character nodes at `/dev/vfio/<group>` and the shared VFIO container at `/dev/vfio/vfio`.

### Supported Google TPU PCI Device IDs (`vendor=0x1ae0`)

| PCI Device ID | Generation | Model Label | Nominal HBM / Chip |
| --- | --- | --- | --- |
| `0x0027` | `v2-v3` | TPU v2/v3 | 16 GiB |
| `0x0056` | `v4lite` | TPU v4lite | 8 GiB |
| `0x005e` | `v4` | TPU v4 | 32 GiB |
| `0x0062` | `v5p` | TPU v5p | 95 GiB |
| `0x0063` | `v5e` | TPU v5e | 16 GiB |
| `0x006e` | `v6e-pf` | TPU v6e (PF) | 32 GiB |
| `0x006f` | `v6e` | TPU v6e (Trillium) | 32 GiB |

---

## 1. Prometheus Telemetry (`ops/tpu-exporter/`)

### Running the Exporter

```bash
# Long-running HTTP server (binds 127.0.0.1:9400 by default; pass --host 0.0.0.0 in pods/containers)
python3 ops/tpu-exporter/tpu_exporter.py --port 9400

# One-shot snapshot (CI / ephemeral Cloud TPU or Colab runs)
python3 ops/tpu-exporter/tpu_exporter.py --once

# Structured JSON snapshot
python3 ops/tpu-exporter/tpu_exporter.py --once --json
```

HTTP endpoints served in server mode:
- `GET /metrics` — Prometheus text exposition format `0.0.4`
- `GET /metrics.json` — Structured JSON telemetry snapshot
- `GET /healthz` — Readiness/liveness probe (`{"status":"ok"}`)

### Exported Metrics

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `freetpu_exporter_up` | gauge | — | Scrape liveness (`1`) |
| `freetpu_tpu_devices_discovered` | gauge | — | Count of `0x1ae0` PCI TPU devices |
| `freetpu_tpu_info` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `vendor`, `model`, `driver`, `iommu_group`, `telemetry_source` | Static device inventory (`1`) |
| `freetpu_tpu_health_status` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` if `vfio-pci` bound and VFIO nodes present (idle chips remain `1`) |
| `freetpu_tpu_pci_enabled` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | PCI `enable` sysfs state (`1` when opened by a VFIO client, `0` when idle) |
| `freetpu_tpu_pid_telemetry_available` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` when active PID reads succeeded, `0` when `/proc/*/fd` was unreadable |
| `freetpu_tpu_hbm_used_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Live HBM allocated bytes (only when verified PCI<->runtime mapping exists) |
| `freetpu_tpu_hbm_total_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Total HBM capacity bytes |
| `freetpu_tpu_hbm_utilization_ratio` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `hbm_used_bytes / hbm_total_bytes` |
| `freetpu_tpu_duty_cycle_percent` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | MXU duty cycle `%` from `TPU Runtime Utilization` |
| `freetpu_tpu_active_pids` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Count of active host PIDs on chip (omitted if unreadable) |
| `freetpu_tpu_process_active` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `pid` | Active process marker (`1`) |
| `freetpu_tpu_runtime_hbm_used_bytes` | gauge | `runtime_chip`, `pci_bdf` | Live HBM bytes keyed by `libtpu` runtime chip ID (`pci_bdf="unmapped"` if no map) |
| `freetpu_tpu_runtime_duty_cycle_percent` | gauge | `runtime_chip`, `pci_bdf` | MXU duty cycle `%` keyed by `libtpu` runtime chip ID |
| `freetpu_tpu_scrape_duration_seconds` | gauge | — | Scrape duration in seconds |

Telemetry source priority:
1. **`tpu-info` CLI**: Parses `TPU Chips` and `TPU Runtime Utilization` tables for live HBM usage, MXU duty cycle `%`, and active PIDs.
2. **Pure `sysfs` + `/proc/<pid>/fd` fallback**: When `tpu-info` is not installed, `tpu-exporter` still reports PCI/VFIO topology, chip health, nominal HBM capacity, and active PIDs holding `/dev/vfio/<group>` file descriptors.

See [`ops/tpu-exporter/prometheus-scrape.yml`](../../ops/tpu-exporter/prometheus-scrape.yml) and [`ops/tpu-exporter/promql-queries.md`](../../ops/tpu-exporter/promql-queries.md) for Prometheus scrape wiring and billing/waste PromQL queries.

---

## 2. Container Passthrough via CNCF CDI (`ops/tpu-cdi/`)

### Generating & Validating `/etc/cdi/freetpu.json`

```bash
# Write /etc/cdi/freetpu.json and validate against CNCF CDI v0.6.0 rules
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /etc/cdi/freetpu.json --validate

# Preview CDI JSON to stdout
python3 ops/tpu-cdi/freetpu_cdi.py generate --stdout --validate

# Validate an existing CDI spec
python3 ops/tpu-cdi/freetpu_cdi.py validate /etc/cdi/freetpu.json
```

Generated CDI devices:
- `freetpu.dev/tpu=0`, `freetpu.dev/tpu=1`, ... — Individual TPU chip `/dev/vfio/<iommu_group>` plus global `/dev/vfio/vfio` (when each chip has its own dedicated IOMMU group).
- `freetpu.dev/tpu=group-<iommu_group>` — Shared IOMMU group allocation unit (emitted when multiple chips share one `/dev/vfio/<iommu_group>` instead of misleading per-chip selectors).
- `freetpu.dev/tpu=all` — All discovered TPU `/dev/vfio/<iommu_group>` nodes (deduplicated) plus `/dev/vfio/vfio`.

### Docker / Podman / containerd Usage

VFIO DMA pinning requires `CAP_IPC_LOCK` and unlimited `RLIMIT_MEMLOCK` configured **before** container creation (CNCF CDI v0.6.0 does not configure capabilities or rlimits, and runtime hooks run too late):

```bash
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(sorted(os.listdir('/dev/vfio')))"
```

Print the exact runtime flags with:

```bash
python3 ops/tpu-cdi/freetpu_cdi.py docker-args --device all
```

### Kubernetes Deployment

First provision the `freetpu-ops-scripts` ConfigMap in `kube-system`:

```bash
kubectl -n kube-system create configmap freetpu-ops-scripts \
  --from-file=ops/tpu-cdi/freetpu_cdi.py \
  --from-file=ops/tpu-exporter/tpu_exporter.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

- Node setup & metrics: [`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml)
- Sample `DeviceClass`, DRA `ResourceClaimTemplate`, and Pod: [`ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml) (requires Kubernetes >= 1.32 with DRA enabled and a cluster DRA driver publishing `ResourceSlice` objects for `freetpu.dev`)

---

## 3. Offline Self-Tests

Both tools include deterministic unit tests backed by synthetic `/sys/bus/pci/devices/*`, `/dev/vfio/*`, `/proc/<pid>/fd/*`, and mock `tpu-info` fixtures (wired into `.github/workflows/ci.yml`):

```bash
python3 ops/tpu-exporter/test_tpu_exporter.py -v
python3 ops/tpu-cdi/test_freetpu_cdi.py -v
```
