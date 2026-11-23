# Cloud TPU Ops Plumbing: Container CDI & Prometheus Telemetry

FreeTPU Pillar 6 brings GPU-grade operational ergonomics (`dcgm-exporter` + `nvidia-container-toolkit` + Kubernetes device plumbing) to VFIO-backed Google Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium) using zero-dependency Python tooling over standard Linux sysfs (`vendor=0x1ae0`) and VFIO (`/dev/vfio/*`).

---

## Architecture Overview

| NVIDIA GPU Stack | FreeTPU Cloud TPU Equivalent | Path |
| --- | --- | --- |
| `nvidia-dcgm-exporter` | `tpu-exporter` (`:9400/metrics`) | [`ops/tpu-exporter/tpu_exporter.py`](../../ops/tpu-exporter/tpu_exporter.py) |
| `nvidia-ctk cdi generate` | `freetpu-cdi generate` (CNCF CDI `v0.6.0` default, `--cdi-version 1.1.0` opt-in) | [`ops/tpu-cdi/freetpu_cdi.py`](../../ops/tpu-cdi/freetpu_cdi.py) |
| `k8s-device-plugin` / GPU Operator | K8s DaemonSet + DRA / CDI Pod Manifests | [`ops/tpu-cdi/manifests/`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) |

VFIO-backed Cloud TPU VMs (`v5e`, `v5p`, `v6e` Trillium) expose each TPU chip as a PCI function (`vendor=0x1ae0`) bound to the standard Linux `vfio-pci` kernel driver, with per-IOMMU-group character nodes at `/dev/vfio/<group>` and the shared VFIO container at `/dev/vfio/vfio`.

> **Cloud TPU v2 / v3 / v4 (`/dev/accel*`) Scope Note:** Google Cloud TPU `v2`, `v3`, and `v4` hosts use Google's `/dev/accel*` character device interface rather than `vfio-pci` + `/dev/vfio/*`. `tpu-exporter` still recognizes `v2/v3` and `v4` PCI IDs for `freetpu_tpu_info` inventory discovery, but `freetpu-cdi` passthrough, VFIO readiness (`freetpu_tpu_health_status`), and `/dev/vfio/*` PID tracking are scoped strictly to VFIO-backed hosts (`v5e`, `v5p`, `v6e` Trillium).

### Supported Google TPU PCI Device IDs (`vendor=0x1ae0`)

| PCI Device ID | Generation | Model Label | Nominal HBM / Chip | Host Interface & Support Scope |
| --- | --- | --- | --- | --- |
| `0x0027` (`subsys 0x004e` / `0x004f`) | `v2` / `v3` | TPU v2 / TPU v3 | 16 GiB / 32 GiB | `/dev/accel*` (PCI inventory only, dual PCI functions grouped per chip; VFIO CDI & health N/A) |
| `0x0056` | `v4lite` | TPU v4lite | 8 GiB | `/dev/accel*` (PCI inventory only; VFIO CDI & health N/A) |
| `0x005e` | `v4` | TPU v4 | 32 GiB | `/dev/accel*` (PCI inventory only; VFIO CDI & health N/A) |
| `0x0062` | `v5p` | TPU v5p | 95 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x0063` | `v5e` | TPU v5e | 16 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x006e` | `v6e-pf` | TPU v6e (PF) | 32 GiB | `vfio-pci` (`/dev/vfio/*`, sysfs/VFIO inventory + CDI; upstream `tpu-info` CLI discovery omits `0x006e`, while `tpu_exporter.py` maps `0x006e` to `TpuChip.V6E` and queries a reachable `libtpu` endpoint via direct gRPC or `tpu_info.metrics` fallback) |
| `0x006f` | `v6e` | TPU v6e (Trillium) | 32 GiB | `vfio-pci` (`/dev/vfio/*`, full exporter + CDI) |
| `0x0075` | `tpu7x` | TPU tpu7x (Ironwood, pending live hardware confirmation `#21`) | 192 GiB | `vfio-pci` (`/dev/vfio/*` or `iommufd`, pending live HW confirmation) |
| `0x0076` | `tpu7x` | TPU tpu7x (Ironwood) | 192 GiB | `vfio-pci` (`/dev/vfio/*` or `iommufd`, matches upstream `tpu-info` Ironwood ID) |

> **TPU v6e PF (`0x006e`) note:** Upstream `tpu-info` CLI discovery recognizes the v6e virtual function (`0x006f`) and omits the physical function (`0x006e`), exiting before querying `libtpu` if only `0x006e` is present. On PF-only `0x006e` hosts, `tpu_exporter.py` maps `0x006e` directly to `TpuChip.V6E` and falls back to the Python `tpu_info.metrics` gRPC client against `--runtime-metrics-addr` (default `127.0.0.1:8431`) when the `tpu-info` Python package is installed, so live HBM and duty-cycle telemetry works on PF-only hosts without reconfiguring PCI devices (while PCI/VFIO inventory, readiness, active PIDs, and `freetpu-cdi` work without `tpu-info`).

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
| `freetpu_tpu_info` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device`, `vendor`, `model`, `driver`, `iommu_group`, `vfio_group_exclusive`, `telemetry_source` | Static device inventory (`1`; `vfio_group_exclusive="true"` only when full `/sys/kernel/iommu_groups/<group>/devices` membership is verified to contain solely this TPU) |
| `freetpu_tpu_health_status` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | `1` if `vfio-pci` bound, IOMMU group is viable (no sibling endpoint bound to a conflicting host driver), and `/dev/vfio` char nodes match kernel sysfs (`0` on non-VFIO `/dev/accel*` hosts; idle viable VFIO chips remain `1`) |
| `freetpu_tpu_pci_enabled` | gauge | `chip`, `pci_bdf`, `device_id`, `generation`, `vfio_device` | PCI enable state (`1` when sysfs `enable` reference count > 0, `0` otherwise; does not by itself identify a VFIO client) |
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

See [`ops/tpu-exporter/prometheus-scrape.yml`](../../ops/tpu-exporter/prometheus-scrape.yml) (trusted node Prometheus) and [`ops/tpu-exporter/promql-queries.md`](../../ops/tpu-exporter/promql-queries.md) for Prometheus scrape wiring and fleet/utilization PromQL queries.

---

## 2. Container Passthrough via CNCF CDI (`ops/tpu-cdi/`)

### Generating, Validating & Reconciling Boot-Scoped `/var/run/cdi/freetpu.json`

`freetpu-cdi` defaults to the boot-scoped CNCF CDI dynamic directory (`/var/run/cdi/freetpu.json`, tmpfs on Linux) so stale IOMMU group assignments never persist across reboots. Avoid persisting static specs in `/etc/cdi` across reboots or PCI/VFIO topology changes. On Podman releases that search only `/etc/cdi` by default (e.g. Podman `5.4.x`), add `cdi_spec_dirs = ["/etc/cdi", "/var/run/cdi"]` under `[engine]` in `/etc/containers/containers.conf` (or `~/.config/containers/containers.conf`) so Podman discovers `/var/run/cdi/freetpu.json` while keeping specs boot-scoped.

```bash
# Recommended on mutable hosts: continuously reconcile /var/run/cdi/freetpu.json
# (rewrites on VFIO minor/topology change so CDI 0.8.0 drops cached devnums; unlinks on drift/error)
sudo python3 ops/tpu-cdi/freetpu_cdi.py sync --output /var/run/cdi/freetpu.json --interval 15

# One-shot boot/topology snapshot (MUST be removed and regenerated on any VFIO/PCI rebind or topology change)
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /var/run/cdi/freetpu.json --validate

# Preview CDI JSON to stdout (dry-run --allow-missing-vfio requires --stdout and never writes to disk)
python3 ops/tpu-cdi/freetpu_cdi.py generate --stdout --validate

# Revalidate an existing CDI spec against live host VFIO group identity and char st_rdev (removing it if stale)
sudo python3 ops/tpu-cdi/freetpu_cdi.py validate /var/run/cdi/freetpu.json --check-host --remove-if-stale
```

Generated CDI devices (VFIO-backed `v5e`, `v5p`, `v6e` Trillium hosts only):
- `freetpu.dev/tpu=pci-<bdf>` (e.g. `freetpu.dev/tpu=pci-0000-00-06-0`) — Stable hardware-backed selector for an individual TPU chip `/dev/vfio/<iommu_group>` plus global `/dev/vfio/vfio` (when each chip has its own dedicated IOMMU group). Because selectors are bound to the physical PCI BDF rather than a positional index, if a chip disappears its selector is retired instead of shifting onto a different chip.
- `freetpu.dev/tpu=group-pci-<bdf1>_<bdf2>` — Shared IOMMU group allocation unit (emitted when multiple TPU chips share one `/dev/vfio/<iommu_group>` instead of misleading per-chip selectors).
- `freetpu.dev/tpu=all` — All discovered TPU `/dev/vfio/<iommu_group>` nodes (deduplicated) plus `/dev/vfio/vfio`.

Safety & runtime guarantees:
- **Full IOMMU group verification & live host revalidation:** `freetpu-cdi` inspects every PCI member of each TPU's IOMMU group (both `/sys/bus/pci/devices/*/iommu_group` and `/sys/kernel/iommu_groups/<group>/devices/*`) and fails closed if any group also contains a non-TPU PCI device, preventing accidental VFIO exposure through any selector (including `all`). Failed regenerations unlink stale specs at `--output`, `validate --check-host --remove-if-stale` verifies all effective global and per-device `hostPath` nodes against live host VFIO groups (rejecting pinned `major`/`minor` numbers, unauthorized global nodes, and unchecked `mounts` or OCI `hooks`), and `sync` continuously reconciles or invalidates `/var/run/cdi/freetpu.json` if group assignments change.
- **Dynamic VFIO device numbers & CDI v0.8.0 pre-rebind withdrawal:** Generated `deviceNodes` intentionally omit static `major`/`minor` numbers, while live observed char device numbers (`st_rdev`) are recorded in `freetpu.dev/vfio-devnums` annotations. Because CDI v0.8.0 (containerd 2.0) resolves omitted `major`/`minor` numbers on **first injection** and caches them in the loaded device entry until the spec file changes or is removed—and because a later spec refresh never revokes device nodes or permissions already injected into running containers—even continuous `freetpu_cdi.py sync` leaves a polling window (default 15 s) and would recreate `/var/run/cdi/freetpu.json` on the next poll if deleted while the old topology is still valid. Before any PCI/VFIO unbind or rebind, operators **must**: (1) stop/drain affected container consumers, (2) stop or pause `freetpu_cdi.py sync`, (3) delete `/var/run/cdi/freetpu.json` and confirm the container runtime has withdrawn the CDI selectors, (4) perform the PCI/VFIO unbind/rebind, and (5) regenerate or resume `freetpu_cdi.py sync`.

### Docker / Podman / containerd Usage

Linux VFIO type1 DMA pinning requires a sufficient `RLIMIT_MEMLOCK` configured **before** container creation (CNCF CDI v0.6.0 does not configure rlimits or capabilities, and runtime hooks run too late). `--ulimit memlock=-1:-1` (or a bounded limit covering the workload's pinned DMA buffers) works without extra capabilities; `--cap-add=IPC_LOCK` (`docker-args --cap-ipc-lock`) is only needed as an optional bypass when `RLIMIT_MEMLOCK` cannot be raised:

```bash
docker run --rm \
  --device=freetpu.dev/tpu=all \
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

- Node setup, opt-in CDI sync & PCI/VFIO inventory metrics (**GKE Standard** / self-managed Kubernetes only; **GKE Autopilot** blocks `hostPath` volumes at admission—use workload-side `tpu_exporter.py --no-pid-scan` on Autopilot): [`ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) — targets all VFIO-backed GKE Standard TPU node types (`tpu-v5-lite-device`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`) via `nodeAffinity` and tolerates `google.com/tpu`, `cloud.google.com/gke-spot`, and `cloud.google.com/gke-queued` (`NoSchedule`) taints. By default, `DaemonSet/freetpu-node-agent` runs with `--no-pid-scan --tpu-info-bin ""` (no `hostPID`, no `CAP_SYS_PTRACE`, no `Unconfined` AppArmor, stock `python:3.12-slim` without `tpu-info`, and loopback-only liveness/readiness on `127.0.0.1:9401`) to expose **PCI/VFIO hardware inventory (`vfio_group_exclusive`) and VFIO readiness only** without publishing CDI specs, paired with a `NetworkPolicy` (`freetpu-node-agent-metrics`) restricting `:9400` ingress to the trusted `monitoring` Prometheus scraper. Because GKE Standard accepts `NetworkPolicy` objects even when policy enforcement is disabled, on shared/multi-tenant clusters (including default inventory-only mode) enforced NetworkPolicy plus admission controls forbidding tenant `hostNetwork: true` pods—or binding `--host 127.0.0.1` behind an authenticating proxy such as `kube-rbac-proxy`—is a prerequisite so tenant workloads cannot starve the exporter's bounded `:9400` scrape worker pool. Host PID telemetry (`freetpu_tpu_active_pids`) is an additional explicit opt-in—follow the prerequisites in [`k8s-tpu-cdi-daemonset.yaml`](../../ops/tpu-cdi/manifests/k8s-tpu-cdi-daemonset.yaml) (enable `hostPID: true`, `CAP_SYS_PTRACE`, `appArmorProfile.type: Unconfined` on GKE, and remove `--no-pid-scan` only after verifying that ingress isolation). CDI publication is opt-in via a separate `DaemonSet/freetpu-cdi-sync` (`--unlink-on-exit`) that schedules ONLY on nodes explicitly labeled `freetpu.dev/cdi-enabled=true` after verifying containerd is unaffected/patched for CVE-2026-53492 (upstream affected: `2.1.0–2.1.8`, `2.2.0–2.2.4`, `2.3.0–2.3.1`; fixed in `2.1.9`, `2.2.5`, `2.3.2`, `>= 2.4.0`; upstream `2.0.x` lacks the affected checkpoint-restore path, though vendor/GKE builds should still be verified for backports) or blocking untrusted container checkpoint restores. On **GKE Standard**, use `freetpu-node-agent` for telemetry and native `google.com/tpu` scheduling (GKE DRA currently supports GPUs only, not TPUs); on **self-managed Kubernetes**, pair published CDI specs with a cluster DRA driver registered for `freetpu.dev`. Because CDI device injection does not set `RLIMIT_MEMLOCK` or capabilities and Kubernetes Pods do not expose `--ulimit`, verify each TPU workload's effective memlock limit on self-managed nodes (raise the runtime's memlock limit, e.g. `LimitMEMLOCK=infinity` in `containerd.service`, or grant an approved `IPC_LOCK` capability to TPU workload containers).

---

## 3. Offline Self-Tests

Both tools include deterministic unit tests backed by synthetic `/sys/bus/pci/devices/*`, `/dev/vfio/*`, `/proc/<pid>/fd/*`, and mock `tpu-info` fixtures (wired into `.github/workflows/ci.yml`):

```bash
python3 ops/tpu-exporter/test_tpu_exporter.py -v
python3 ops/tpu-cdi/test_freetpu_cdi.py -v
```
