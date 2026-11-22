# Cloud TPU Ops Plumbing: Container CDI & Prometheus Telemetry

FreeTPU Pillar 6 brings GPU-grade operational ergonomics (`dcgm-exporter` + `nvidia-container-toolkit` + Kubernetes device plumbing) to VFIO-backed Google Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium) using zero-dependency Python tooling over standard Linux sysfs (`vendor=0x1ae0`) and VFIO (`/dev/vfio/*`).

---

## Architecture Overview

| NVIDIA GPU Stack | FreeTPU Cloud TPU Equivalent | Path |
| --- | --- | --- |
| `nvidia-dcgm-exporter` | `tpu-exporter` (`:9400/metrics`) | [`ops/tpu-exporter/tpu_exporter.py`](../../ops/tpu-exporter/tpu_exporter.py) |
| `nvidia-ctk cdi generate` | `freetpu-cdi generate` (CNCF CDI `v0.6.0`) | [`ops/tpu-cdi/freetpu_cdi.py`](../../ops/tpu-cdi/freetpu_cdi.py) |
| `k8s-device-plugin` / GPU Operator | K8s DaemonSet + DRA / CDI Pod Manifests | [`ops/tpu-cdi/manifests/`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) |

VFIO-backed Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium) expose each TPU chip as a PCI function (`vendor=0x1ae0`) bound to the standard Linux `vfio-pci` kernel driver, with per-IOMMU-group character nodes at `/dev/vfio/<group>` and the shared VFIO container at `/dev/vfio/vfio`.

> **Cloud TPU v2 / v3 / v4 (`/dev/accel*`) Scope Note:** Google Cloud TPU `v2`, `v3`, and `v4` hosts use Google's `/dev/accel*` character device interface rather than `vfio-pci` + `/dev/vfio/*`. `tpu-exporter` still recognizes `v2/v3` and `v4` PCI IDs for `freetpu_tpu_info` inventory discovery, but `freetpu-cdi` passthrough, VFIO readiness (`freetpu_tpu_health_status`), and `/dev/vfio/*` PID tracking are scoped strictly to VFIO-backed hosts (`v5e`, `v5p`, `v6e` Trillium).

### Supported Google TPU PCI Device IDs (`vendor=0x1ae0`)

| PCI Device ID | Generation | Model Label | Nominal HBM / Chip | Host Interface & Support Scope |
| --- | --- | --- | --- | --- |
| `0x0027` | `v2-v3` | TPU v2/v3 | 16 GiB | `/dev/accel*` (PCI inventory only; VFIO CDI & health N/A) |
| `0x0056` | `v4lite` | TPU v4lite | 8 GiB | `/dev/accel*` (PCI inventory only; VFIO CDI & health N/A) |
| `0x005e` | `v4` | TPU v4 | 32 GiB | `/dev/accel*` (PCI inventory only; VFIO CDI & health N/A) |
| `0x0062` | `v5p` | TPU v5p | 95 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x0063` | `v5e` | TPU v5e | 16 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x006e` | `v6e-pf` | TPU v6e (PF) | 32 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x006f` | `v6e` | TPU v6e (Trillium) | 32 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |

---

## 1. Prometheus Telemetry (`ops/tpu-exporter/`)

### Running the Exporter

```bash
# Long-running HTTP server (binds 127.0.0.1:9400 by default; pass --host 0.0.0.0 in pods/containers
# and --health-port 9401 to isolate liveness/readiness probes from scrape traffic saturation)
python3 ops/tpu-exporter/tpu_exporter.py --port 9400 --health-port 9401

# One-shot snapshot (CI / ephemeral Cloud TPU or Colab runs)
python3 ops/tpu-exporter/tpu_exporter.py --once

# Structured JSON snapshot
python3 ops/tpu-exporter/tpu_exporter.py --once --json
```

HTTP endpoints served in server mode:
- `GET /metrics` — Prometheus text exposition format `0.0.4`
- `GET /metrics.json` — Structured JSON telemetry snapshot
- `GET /healthz`, `GET /readyz` — Readiness/liveness probe (`{"status":"ok"}`); pass `--health-port` (e.g. `9401`) to serve probes on a dedicated listener completely independent of `/metrics` scrape saturation.

### Exported Metrics

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `freetpu_exporter_up` | gauge | — | Scrape liveness (`1`) |
| `freetpu_tpu_devices_discovered` | gauge | — | Count of `0x1ae0` PCI TPU devices |
| `freetpu_tpu_info` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `vendor`, `model`, `driver`, `iommu_group`, `telemetry_source` | Static device inventory (`1`) |
| `freetpu_tpu_health_status` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` if `vfio-pci` bound and `/dev/vfio` nodes present on VFIO-backed hosts (`0` on non-VFIO `/dev/accel*` hosts; idle VFIO chips remain `1`) |
| `freetpu_tpu_pci_enabled` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | PCI `enable` sysfs state (`1` when opened by a VFIO client, `0` when idle) |
| `freetpu_tpu_pid_telemetry_available` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` when active PID reads succeeded, `0` when `/proc/*/fd` was unreadable |
| `freetpu_tpu_hbm_used_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Live HBM allocated bytes (only when verified PCI<->runtime mapping and usable `tpu-info` reading exist) |
| `freetpu_tpu_hbm_total_bytes` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Total HBM capacity bytes |
| `freetpu_tpu_hbm_utilization_ratio` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `hbm_used_bytes / hbm_total_bytes` |
| `freetpu_tpu_duty_cycle_percent` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | MXU duty cycle `%` from `TPU Device Status` or `TPU Runtime Utilization` (omitted when unavailable/unparsed) |
| `freetpu_tpu_active_pids` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | Count of active host PIDs on chip (omitted if unreadable) |
| `freetpu_tpu_process_active` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `pid` | Active process marker (`1`) |
| `freetpu_tpu_runtime_hbm_used_bytes` | gauge | `runtime_chip`, `pci_bdf` | Live HBM bytes keyed by `libtpu` runtime chip ID (`pci_bdf="unmapped"` if no map) |
| `freetpu_tpu_runtime_duty_cycle_percent` | gauge | `runtime_chip`, `pci_bdf` | MXU duty cycle `%` keyed by `libtpu` runtime chip ID |
| `freetpu_tpu_scrape_duration_seconds` | gauge | — | Scrape duration in seconds |

Telemetry source priority:
1. **`tpu-info` CLI**: Parses both the modern `tpu-info` dashboard (`TPU Device Status` and `TPU Process Monitor`) and classic (`TPU Chips` and `TPU Runtime Utilization`) tables for live HBM usage, TensorCore/MXU duty cycle `%`, and active PIDs. When `tpu-info` reports `N/A` or output cannot be parsed, HBM usage and duty cycle are left unset rather than fabricated as `0`.
2. **Pure `sysfs` + `/proc/<pid>/fd` fallback**: When `tpu-info` is not installed, `tpu-exporter` still reports PCI/VFIO topology, VFIO chip readiness, nominal HBM capacity, and active PIDs holding `/dev/vfio/<group>` file descriptors.

See [`ops/tpu-exporter/prometheus-scrape.yml`](../../ops/tpu-exporter/prometheus-scrape.yml) and [`ops/tpu-exporter/promql-queries.md`](../../ops/tpu-exporter/promql-queries.md) for Prometheus scrape wiring and billing/waste PromQL queries.

---

## 2. Container Passthrough via CNCF CDI (`ops/tpu-cdi/`)

### Generating, Validating & Reconciling Boot-Scoped `/var/run/cdi/freetpu.json`

`freetpu-cdi` defaults to the boot-scoped CNCF CDI dynamic directory (`/var/run/cdi/freetpu.json`, tmpfs on Linux) so stale IOMMU group assignments never persist across reboots. Avoid persisting static specs in `/etc/cdi` across reboots or PCI/VFIO topology changes.

```bash
# Write boot-scoped /var/run/cdi/freetpu.json and validate against CNCF CDI v0.6.0 rules
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /var/run/cdi/freetpu.json --validate

# Preview CDI JSON to stdout (dry-run --allow-missing-vfio requires --stdout and never writes to disk)
python3 ops/tpu-cdi/freetpu_cdi.py generate --stdout --validate

# Revalidate an existing CDI spec against live host VFIO group identity (removing it if stale)
sudo python3 ops/tpu-cdi/freetpu_cdi.py validate /var/run/cdi/freetpu.json --check-host --remove-if-stale

# Continuously reconcile /var/run/cdi/freetpu.json on live VFIO topology changes (invalidating on drift/error)
sudo python3 ops/tpu-cdi/freetpu_cdi.py sync --output /var/run/cdi/freetpu.json --interval 15
```

Generated CDI devices (VFIO-backed `v5e`, `v5p`, `v6e` Trillium hosts only):
- `freetpu.dev/tpu=pci-<bdf>` (e.g. `freetpu.dev/tpu=pci-0000-00-06-0`) — Stable hardware-backed selector for an individual TPU chip `/dev/vfio/<iommu_group>` plus global `/dev/vfio/vfio` (when each chip has its own dedicated IOMMU group). Because selectors are bound to the physical PCI BDF rather than a positional index, if a chip disappears its selector is retired instead of shifting onto a different chip.
- `freetpu.dev/tpu=group-pci-<bdf1>_<bdf2>` — Shared IOMMU group allocation unit (emitted when multiple TPU chips share one `/dev/vfio/<iommu_group>` instead of misleading per-chip selectors).
- `freetpu.dev/tpu=all` — All discovered TPU `/dev/vfio/<iommu_group>` nodes (deduplicated) plus `/dev/vfio/vfio`.

Safety & runtime guarantees:
- **Full IOMMU group verification & live host revalidation:** `freetpu-cdi` inspects every PCI member of each TPU's IOMMU group (both `/sys/bus/pci/devices/*/iommu_group` and `/sys/kernel/iommu_groups/<group>/devices/*`) and fails closed if any group also contains a non-TPU PCI device, preventing accidental VFIO exposure through any selector (including `all`). Failed regenerations unlink stale specs at `--output`, `validate --check-host --remove-if-stale` verifies all effective global and per-device `hostPath` nodes against live host VFIO groups (rejecting pinned `major`/`minor` numbers, unauthorized global nodes, and unchecked `mounts` or OCI `hooks`), and `sync` continuously reconciles or invalidates `/var/run/cdi/freetpu.json` if group assignments change.
- **Dynamic VFIO device numbers & CDI cache refresh:** Generated `deviceNodes` intentionally omit `major` and `minor` so containerd/CDI resolves live device numbers from `hostPath` at container start, while live observed char device numbers are recorded in `freetpu.dev/vfio-devnums` annotations so a driver rebind that changes dynamic VFIO minors updates the spec on disk and triggers containerd 2.0 / CDI 0.8.0 to refresh its device cache.

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

- Node setup, CDI sync & inventory/PID metrics: [`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) — targets all VFIO-backed GKE TPU node types (`tpu-v5-lite-device`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`) via `nodeAffinity`, tolerates `google.com/tpu:NoSchedule`, continuously reconciles `/var/run/cdi/freetpu.json` via `freetpu-cdi-sync`, and exposes **PCI/VFIO inventory and hostPID active-PID metrics only** (stock `python:3.12-slim` without `tpu-info`, with liveness/readiness isolated on `:9401`, and `SYS_PTRACE` + `appArmorProfile.type: Unconfined` required on GKE so default AppArmor does not block cross-profile `/proc/<pid>/fd` reads).
- Live HBM + MXU duty-cycle telemetry in Kubernetes (workload sidecar): [`ops/tpu-cdi/manifests/k8s-tpu-workload-telemetry-sidecar.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-workload-telemetry-sidecar.yaml) — because `libtpu` serves runtime metrics on `localhost:8431` inside each workload Pod's network namespace, build `ops/tpu-exporter/Dockerfile` with `--build-arg INSTALL_TPU_INFO=1` and run it as a native sidecar (`initContainers` with `restartPolicy: Always`) inside the TPU workload Pod.
- Sample `DeviceClass`, DRA `ResourceClaimTemplate`, and Pod (stable `resource.k8s.io/v1` for Kubernetes >= 1.34 and `containerd >= 2.0`, with `google.com/tpu:NoSchedule` toleration): [`ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-pod-cdi.yaml) (requires a cluster DRA driver publishing `ResourceSlice` objects for `freetpu.dev`; `containerd >= 2.0` is required for both CDI v0.6.0 `annotations` and kubelet CRI `CDIDevices` injection).
- Legacy Kubernetes 1.32–1.33 DRA sample (`resource.k8s.io/v1beta1`, `containerd >= 2.0`): [`ops/tpu-cdi/manifests/k8s-tpu-pod-cdi-v1beta1.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-pod-cdi-v1beta1.yaml) (requires `--runtime-config=resource.k8s.io/v1beta1=true` and `DynamicResourceAllocation=true`).

---

## 3. Offline Self-Tests

Both tools include deterministic unit tests backed by synthetic `/sys/bus/pci/devices/*`, `/dev/vfio/*`, `/proc/<pid>/fd/*`, and mock `tpu-info` fixtures (wired into `.github/workflows/ci.yml`):

```bash
python3 ops/tpu-exporter/test_tpu_exporter.py -v
python3 ops/tpu-cdi/test_freetpu_cdi.py -v
```
