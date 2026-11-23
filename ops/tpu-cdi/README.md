# `freetpu-cdi` — Open CNCF CDI Toolkit for Cloud TPUs

`freetpu_cdi.py` is an open, zero-dependency CNCF Container Device Interface (CDI) spec generator for VFIO-backed Google Cloud TPUs (`v5e`, `v5p`, `v6e` Trillium, and `tpu7x` Ironwood). By default it emits **CDI `v0.6.0`** (the minimum version needed by all generated fields and compatible with unaffected/patched containerd releases `2.0.x`, `2.1.x >= 2.1.9`, `2.2.x >= 2.2.5`, `2.3.x >= 2.3.2`, or `>= 2.4.0`), with `--cdi-version 1.1.0` available as an explicit opt-in. It scans `/sys/bus/pci/devices/*` (`vendor=0x1ae0`), preserves legacy VFIO (`/dev/vfio/vfio` + `/dev/vfio/<iommu_group>`) as the default whenever usable (with explicit `--vfio-mode iommufd` / `cdev` opt-in and pure-`iommufd` fallback using verified `<pci>/vfio-dev/vfioN` sysfs associations), verifies full IOMMU group isolation and groups sibling chips sharing an IOMMU group into a single allocation unit in both modes, omits static `major`/`minor` numbers from `deviceNodes` while tracking live char device numbers in `freetpu.dev/vfio-devnums` annotations, groups multi-root hosts (e.g. `v6e-8` `0000:00` vs `0000:80`, resolved via sysfs PCI host-bridge ancestry) into capacity-pinned NUMA/PCIe-aware sub-host slices (emitting persistent domain-qualified `numa0-pcie-0000-00-4`/`numa1-pcie-0000-80-4` whenever NUMA locality is known — or `pcie-0000-00-4`/`pcie-0000-80-4` when NUMA is absent — plus verified range aliases `0-3`/`4-7`, never emitting scan-dependent shorter aliases that could rebind across outages, and withdrawing degraded slices whose topology is unsupported for the generation, such as 2-chip or 3-chip `v6e` slices), preserves workload-provided `libtpu` environment variables by default (`--network-mode host`, safe for multi-host slices) with opt-in `--single-host-env` / `--network-mode isolated` overrides (`MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353`, `TPU_HOST_BOUNDS=1,1,1`, `TPU_WORKER_HOSTNAMES=localhost`, `TPU_WORKER_ID=0`, `TPU_SKIP_MDS_QUERY=true`, plus physical-chip-derived `TPU_CHIPS_PER_HOST_BOUNDS` and `TPU_ACCELERATOR_TYPE` on supported full-allocation slice/`all` selectors — and verified `TPU_VISIBLE_CHIPS` / `TPU_VISIBLE_DEVICES` when runtime ordinals are verified via sysfs `chip_id` or `--chip-id-map`, with `--omit-visible-env` available for multi-process launchers like PyTorch/XLA `xmp.spawn()` that assign per-rank visibility via `os.environ.setdefault`, omitting visibility/bounds/accelerator-type on composable `pci-*`/`group-pci-*` selectors), and emits a boot-scoped spec at `/var/run/cdi/freetpu.json` (tmpfs, cleared across reboots).

> **Containerd security prerequisite (CVE-2026-53492 / GHSA-33vj-92qq-66hc):** Before publishing `/var/run/cdi/freetpu.json` on **any** host (whether manually via `generate`/`sync` or via Kubernetes DaemonSet), verify containerd is **not** an affected release (`2.1.0–2.1.8`, `2.2.0–2.2.4`, `2.3.0–2.3.1`, or vendor/GKE builds carrying checkpoint-restore backports — fixed in `2.1.9`, `2.2.5`, `2.3.2`, `>= 2.4.0`; upstream `2.0.x` lacks the affected checkpoint-restore path) **or** confirm untrusted container checkpoint restores are disabled/blocked. On affected releases with checkpoint restore allowed, any published CDI spec can expose TPU devices (including `all`) outside Kubernetes resource allocation.

> **Scope note:** Legacy `/dev/accel*` Cloud TPU hosts (e.g. Cloud TPU `v2`, `v3`, `v4` using Google's `accel` driver instead of `vfio-pci`) are not supported by VFIO CDI passthrough.

> **CDI v0.8.0 (containerd 2.0) rebind safety:** CDI v0.8.0 resolves omitted `major`/`minor` device numbers on **first injection** and caches them in the loaded device entry until the spec file on disk changes or is removed, and refreshing a spec never revokes device nodes or permissions already injected into running containers. Even when `freetpu_cdi.py sync` is running, its polling interval (default 15 s) leaves a window where a reassigned VFIO minor could still be injected from CDI's cache—and if `sync` keeps running while you delete `/var/run/cdi/freetpu.json`, the next poll will immediately recreate the file while the old topology is still valid. Before any PCI/VFIO unbind or rebind, operators **must** follow this sequence: (1) stop/drain affected container consumers, (2) stop or pause the `freetpu_cdi.py sync` reconciler, (3) delete `/var/run/cdi/freetpu.json` and confirm the container runtime has withdrawn the CDI selectors, (4) perform the PCI/VFIO unbind/rebind, and (5) regenerate or resume `freetpu_cdi.py sync`.

## Quickstart

```bash
# 1. Recommended on mutable hosts: continuously reconcile /var/run/cdi/freetpu.json
#    (atomically rewrites on VFIO minor/topology change so CDI drops cached devnums; removes if unsafe)
sudo python3 ops/tpu-cdi/freetpu_cdi.py sync --output /var/run/cdi/freetpu.json --interval 15

# 2. One-shot boot/topology snapshot (MUST be removed and regenerated on any VFIO/PCI rebind or topology change)
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /var/run/cdi/freetpu.json --validate

# 3. Preview to stdout without writing /var/run/cdi (dry-run --allow-missing-vfio requires --stdout)
python3 ops/tpu-cdi/freetpu_cdi.py generate --stdout --validate

# 4. Validate an existing CDI spec against CDI v0.6.0, live host VFIO group identity, and live st_rdev
sudo python3 ops/tpu-cdi/freetpu_cdi.py validate /var/run/cdi/freetpu.json --check-host --remove-if-stale
```

## Docker / Podman / nerdctl Passthrough

With boot-scoped `/var/run/cdi/freetpu.json` generated (and CDI enabled in Docker, Podman, or unaffected/patched containerd `2.0.x` / `2.1.x >= 2.1.9` / `2.2.x >= 2.2.5` / `2.3.x >= 2.3.2` / `>= 2.4.0`):

> **Podman CDI search path:** Affected Podman releases (including Podman `5.4.x`) default `cdi_spec_dirs` to `["/etc/cdi"]` only, so they will not discover `/var/run/cdi/freetpu.json` until `/var/run/cdi` is added. Add `cdi_spec_dirs = ["/etc/cdi", "/var/run/cdi"]` under `[engine]` in `/etc/containers/containers.conf` (or `~/.config/containers/containers.conf`) to keep the TPU spec boot-scoped on tmpfs instead of persisting stale specs under `/etc/cdi`.

```bash
# Attach all TPU chips on the host with a sufficient VFIO memlock limit
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"

# Opt in to single-host isolated env overrides so a NUMA-aligned 4-chip slice on a v6e-8
# runs WITHOUT --net=host (injects MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353,
# TPU_HOST_BOUNDS=1,1,1, TPU_WORKER_HOSTNAMES=localhost, TPU_WORKER_ID=0,
# TPU_SKIP_MDS_QUERY=true, TPU_CHIPS_PER_HOST_BOUNDS=2,2,1, and
# TPU_ACCELERATOR_TYPE=v6e-4 on supported slice/all selectors; composable `pci-*`
# selectors omit bounds/accelerator-type so multi-selector workloads set them explicitly).
# On hosts without sysfs `chip_id` files, pass `--chip-id-map` (JSON object or file
# mapping PCI BDF -> verified runtime chip ID) so CDI also injects verified
# TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES (and stable `0-3`/`4-7` range aliases).
# For multi-process PyTorch/XLA workloads on a sub-host slice, add `--omit-visible-env`
# so CDI omits the full-slice visibility mask while keeping isolated networking and
# topology bounds — NOTE that `--omit-visible-env` alone is NOT sufficient with stock
# `xmp.spawn()`, which scans container `/sys/bus/pci/devices` (seeing all 8 host TPUs
# on a `v6e-8`) and assigns zero-based local ranks `0..N-1`; use a slice-aware launcher
# that sets BOTH the worker process count (`nprocs` / `PJRT_LOCAL_PROCESS_COUNT=4`) AND
# each rank's actual assigned chip ID (`TPU_VISIBLE_CHIPS=<chip_id>`, e.g. `4`,`5`,`6`,`7`
# on the upper slice):
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate \
  --single-host-env \
  --chip-id-map /etc/freetpu/chip-id-map.json \
  --output /var/run/cdi/freetpu.json --validate

docker run --rm \
  --device=freetpu.dev/tpu=numa0-pcie-0000-00-4 \
  --ulimit memlock=-1:-1 \
  my-jax-tpu-image:latest

# Or, if generating without sysfs `chip_id` and without `--chip-id-map`, set the
# verified visibility masks explicitly on the workload container:
docker run --rm \
  --device=freetpu.dev/tpu=numa0-pcie-0000-00-4 \
  --ulimit memlock=-1:-1 \
  -e TPU_VISIBLE_CHIPS=0,1,2,3 \
  -e TPU_VISIBLE_DEVICES=0,1,2,3 \
  my-jax-tpu-image:latest

# Default generation preserves workload-provided libtpu env vars for multi-host slices;
# pass --vfio-mode iommufd for Linux 6.8+ cdev or --cdi-version 1.1.0 for CDI 1.1.0 runtimes:
python3 ops/tpu-cdi/freetpu_cdi.py generate --output /var/run/cdi/freetpu.json --validate
```

Linux VFIO type1 DMA pinning requires a sufficient `RLIMIT_MEMLOCK` configured **before** OCI container startup (`docker run` / `podman run`), because CNCF CDI does not configure rlimits or capabilities and an OCI `createContainer` hook executes after `runc` loads process settings. `--ulimit memlock=-1:-1` (or a bounded limit covering the workload's pinned DMA buffers) works without extra Linux capabilities; `--cap-add=IPC_LOCK` is only needed as an optional bypass (`docker-args --cap-ipc-lock`) when the runtime cannot raise `RLIMIT_MEMLOCK`. Use `python3 ops/tpu-cdi/freetpu_cdi.py docker-args` to emit the recommended flags:

```bash
docker run --rm \
  $(python3 ops/tpu-cdi/freetpu_cdi.py docker-args --device all) \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"
```

Note on IOMMU groups & stable selectors: single-function selectors use stable PCI BDF names (`freetpu.dev/tpu=pci-<bdf>`) so a disappearing chip retires its selector instead of reassigning it to a different chip (annotated as `freetpu.dev/allocation-unit=single-chip` for 1-function-per-chip generations such as `v5e`/`v5p`/`v6e`, or `single-chiplet` for multi-function generations such as `tpu7x` Ironwood so individual chiplets are never counted as full physical chips). If multiple TPU chips share one `/dev/vfio/<iommu_group>` node, `freetpu-cdi` exposes that group strictly as a single allocation unit (`freetpu.dev/tpu=group-pci-<bdf1>_<bdf2>`) plus `freetpu.dev/tpu=all`. If any TPU's IOMMU group also contains a non-TPU PCI device, `freetpu-cdi` fails closed rather than publishing any selector (including `all`) that would expose that non-TPU device. Live VFIO char `major:minor` numbers are tracked in `freetpu.dev/vfio-devnums` annotations (while omitted from `deviceNodes`) so `sync` detects unexpected drift and rewrites `/var/run/cdi/freetpu.json` to drop CDI v0.8.0's cached device numbers; before any planned VFIO/PCI unbind or rebind, always stop affected consumers, stop/pause `sync`, delete `/var/run/cdi/freetpu.json` and confirm runtime selector withdrawal, perform the rebind, and then regenerate or resume `sync`.

## Kubernetes Manifests

> **Prerequisite (GKE Standard / self-managed clusters only):** The node DaemonSets below require `hostPath` mounts (`/sys/...`, `/dev/vfio`, `/var/run/cdi`). They require **GKE Standard** (or a self-managed Kubernetes cluster) and are **not** compatible with **GKE Autopilot**, whose admission policy blocks `hostPath` volumes. On GKE Autopilot, use workload-side `tpu_exporter.py --no-pid-scan` telemetry (sharing `127.0.0.1:8431` with `libtpu`) instead of host DaemonSets.

First create the `freetpu-ops-scripts` ConfigMap in `kube-system`:

```bash
kubectl -n kube-system create configmap freetpu-ops-scripts \
  --from-file=ops/tpu-cdi/freetpu_cdi.py \
  --from-file=ops/tpu-exporter/tpu_exporter.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

- [manifests/k8s-tpu-dra-example.yaml](manifests/k8s-tpu-dra-example.yaml) — Starter Kubernetes DRA `DeviceClass`, `ResourceSlice` (v6e-8 `0000:00` NUMA 0 vs `0000:80` NUMA 1), `ResourceClaimTemplate`, and network-isolated 4-chip `Pod` (no `hostNetwork: true`).
- [manifests/k8s-tpu-cdi-daemonset.yaml](manifests/k8s-tpu-cdi-daemonset.yaml) — Node-level DaemonSets targeting VFIO-backed GKE TPU node pools (`tpu-v5-lite-device`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`): `freetpu-node-agent` runs by default with `--no-pid-scan --tpu-info-bin ""` (no `hostPID`, no `CAP_SYS_PTRACE`, no `Unconfined` AppArmor) to serve **PCI/VFIO hardware inventory (`vfio_group_exclusive`) and VFIO readiness only** on `:9400/metrics` (with isolated loopback liveness/readiness probes on `127.0.0.1:9401`, without publishing CDI specs), plus a `NetworkPolicy` (`freetpu-node-agent-metrics`) restricting `:9400` ingress to the trusted `monitoring` Prometheus scraper. Because GKE Standard accepts `NetworkPolicy` objects even when policy enforcement is disabled, on shared/multi-tenant clusters (including default inventory-only mode) enforced NetworkPolicy plus admission controls forbidding tenant `hostNetwork: true` pods—or binding `--host 127.0.0.1` behind an authenticating proxy such as `kube-rbac-proxy`—is a prerequisite so tenant workloads cannot starve the exporter's bounded `:9400` scrape worker pool. Host PID telemetry (`freetpu_tpu_active_pids`) is an additional explicit opt-in: enable `hostPID: true`, `CAP_SYS_PTRACE`, `appArmorProfile.type: Unconfined` (on GKE), and remove `--no-pid-scan` only after verifying that ingress isolation. A separate opt-in `freetpu-cdi-sync` DaemonSet (`--unlink-on-exit`) publishes boot-scoped `/var/run/cdi/freetpu.json` ONLY on nodes explicitly labeled `freetpu.dev/cdi-enabled=true` after verifying containerd is not an affected CVE-2026-53492 release (upstream affected: `2.1.0–2.1.8`, `2.2.0–2.2.4`, `2.3.0–2.3.1`; fixed in `2.1.9`, `2.2.5`, `2.3.2`, `>= 2.4.0`; upstream `2.0.x` lacks the affected checkpoint-restore path, though vendor/GKE builds should still be verified for backports) or blocking untrusted checkpoint restores. On **GKE Standard**, `freetpu-node-agent` provides host PCI/VFIO telemetry (with opt-in `freetpu-cdi-sync` available on explicitly labeled nodes, while GKE schedules TPUs natively via `google.com/tpu` rather than DRA—GKE DRA currently supports GPUs only). On **self-managed Kubernetes** clusters, pair published CDI specs with a cluster DRA driver registered for `freetpu.dev`. Because CDI device injection does not configure `RLIMIT_MEMLOCK` or capabilities and Kubernetes Pods have no `--ulimit` field, also verify each TPU workload's effective memlock limit before running VFIO workloads—on self-managed nodes inheriting a restrictive systemd memlock limit, raise the container runtime's memlock limit (e.g. `LimitMEMLOCK=infinity` in `containerd.service` or containerd's default OCI spec) or grant an approved `IPC_LOCK` capability to TPU workload containers.
