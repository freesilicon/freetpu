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
# Long-running HTTP server (Prometheus text exposition 0.0.4 on :9400/metrics)
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
| `freetpu_tpu_health_status` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` if `vfio-pci` bound, VFIO nodes present, PCI enabled |
| `freetpu_tpu_hbm_used_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Live HBM allocated bytes |
| `freetpu_tpu_hbm_total_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Total HBM capacity bytes |
| `freetpu_tpu_hbm_utilization_ratio` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `hbm_used_bytes / hbm_total_bytes` |
| `freetpu_tpu_duty_cycle_percent` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | MXU / TensorCore duty cycle `%` |
| `freetpu_tpu_active_pids` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Count of active host PIDs on chip |
| `freetpu_tpu_process_active` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `pid` | Active process marker (`1`) |
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
- `freetpu.dev/tpu=0`, `freetpu.dev/tpu=1`, ... — Individual TPU chip `/dev/vfio/<iommu_group>` plus global `/dev/vfio/vfio`.
- `freetpu.dev/tpu=all` — All discovered TPU `/dev/vfio/<iommu_group>` nodes (deduplicated) plus `/dev/vfio/vfio`.

### Docker / Podman / containerd Usage

VFIO DMA pinning requires `CAP_IPC_LOCK` and unlimited `RLIMIT_MEMLOCK` inside the container:

```bash
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(sorted(os.listdir('/dev/vfio')))"
```

For zero-flag `docker run --device=freetpu.dev/tpu=all`, install `freetpu_cdi.py` at `/usr/local/bin/freetpu-cdi` and pass `--with-oci-hook /usr/local/bin/freetpu-cdi` during `generate` so `createContainer` automatically injects `CAP_IPC_LOCK` and `RLIMIT_MEMLOCK=-1`.

### Kubernetes Deployment

- Node setup & metrics: [`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml)
- Sample workload pod / DRA claim: [`ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml)

---

## 3. Offline Self-Tests

Both tools include deterministic unit tests backed by synthetic `/sys/bus/pci/devices/*`, `/dev/vfio/*`, `/proc/<pid>/fd/*`, and mock `tpu-info` fixtures (wired into `.github/workflows/ci.yml`):

```bash
python3 ops/tpu-exporter/test_tpu_exporter.py -v
python3 ops/tpu-cdi/test_freetpu_cdi.py -v
```
