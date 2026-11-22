# `freetpu-cdi` — Open CNCF CDI Toolkit for Cloud TPUs

`freetpu_cdi.py` is an open, zero-dependency CNCF Container Device Interface (CDI v0.6.0) spec generator for VFIO-backed Google Cloud TPUs (`v5e`, `v5p`, `v6e` Trillium bound to `vfio-pci`, analogous to `nvidia-ctk cdi generate`). It scans `/sys/bus/pci/devices/*` (`vendor=0x1ae0`) and `/dev/vfio/*`, verifies full IOMMU group isolation, omits static `major`/`minor` numbers from `deviceNodes` while tracking live char device numbers in `freetpu.dev/vfio-devnums` annotations, and emits a boot-scoped spec at `/var/run/cdi/freetpu.json` (tmpfs, cleared across reboots) so Docker, Podman, containerd (`2.0.x`, `2.1.x >= 2.1.9`, `2.2.x >= 2.2.5`, `2.3.x >= 2.3.2`, or `>= 2.4.0` — excluding CVE-2026-53492 affected `2.1.0–2.1.8`, `2.2.0–2.2.4`, `2.3.0–2.3.1`), and Kubernetes can attach TPU chips cleanly without privileged host-network/host-device hacks.

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

```bash
# Attach all TPU chips on the host with a sufficient VFIO memlock limit
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"

# Attach a single TPU chip by its stable PCI BDF selector (never shifts if another chip disappears).
# On multi-chip VMs, also configure libtpu for the selected chip subset before initializing JAX/libtpu
# (set TPU_VISIBLE_CHIPS to the host chip index corresponding to the selected VFIO group, plus single-chip bounds):
docker run --rm \
  --device=freetpu.dev/tpu=pci-0000-00-06-0 \
  --ulimit memlock=-1:-1 \
  -e TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
  -e TPU_HOST_BOUNDS=1,1,1 \
  -e TPU_VISIBLE_CHIPS=0 \
  my-jax-tpu-image:latest
```

Linux VFIO type1 DMA pinning requires a sufficient `RLIMIT_MEMLOCK` configured **before** OCI container startup (`docker run` / `podman run`), because CNCF CDI v0.6.0 does not configure rlimits or capabilities and an OCI `createContainer` hook executes after `runc` loads process settings. `--ulimit memlock=-1:-1` (or a bounded limit covering the workload's pinned DMA buffers) works without extra Linux capabilities; `--cap-add=IPC_LOCK` is only needed as an optional bypass (`docker-args --cap-ipc-lock`) when the runtime cannot raise `RLIMIT_MEMLOCK`. Use `python3 ops/tpu-cdi/freetpu_cdi.py docker-args` to emit the recommended flags:

```bash
docker run --rm \
  $(python3 ops/tpu-cdi/freetpu_cdi.py docker-args --device all) \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"
```

Note on IOMMU groups & stable selectors: single-chip selectors use stable PCI BDF names (`freetpu.dev/tpu=pci-<bdf>`) so a disappearing chip retires its selector instead of reassigning it to a different chip. If multiple TPU chips share one `/dev/vfio/<iommu_group>` node, `freetpu-cdi` exposes that group strictly as a single allocation unit (`freetpu.dev/tpu=group-pci-<bdf1>_<bdf2>`) plus `freetpu.dev/tpu=all`. If any TPU's IOMMU group also contains a non-TPU PCI device, `freetpu-cdi` fails closed rather than publishing any selector (including `all`) that would expose that non-TPU device. Live VFIO char `major:minor` numbers are tracked in `freetpu.dev/vfio-devnums` annotations (while omitted from `deviceNodes`) so `sync` detects unexpected drift and rewrites `/var/run/cdi/freetpu.json` to drop CDI v0.8.0's cached device numbers; before any planned VFIO/PCI unbind or rebind, always stop affected consumers, stop/pause `sync`, delete `/var/run/cdi/freetpu.json` and confirm runtime selector withdrawal, perform the rebind, and then regenerate or resume `sync`.

## Kubernetes Manifests

> **Prerequisite (GKE Standard / self-managed clusters only):** The node DaemonSets below require `hostPath` mounts (`/sys/...`, `/dev/vfio`, `/var/run/cdi`). They require **GKE Standard** (or a self-managed Kubernetes cluster) and are **not** compatible with **GKE Autopilot**, whose admission policy blocks `hostPath` volumes. On GKE Autopilot, use workload-side `tpu_exporter.py --no-pid-scan` telemetry (sharing `127.0.0.1:8431` with `libtpu`) instead of host DaemonSets.

First create the `freetpu-ops-scripts` ConfigMap in `kube-system`:

```bash
kubectl -n kube-system create configmap freetpu-ops-scripts \
  --from-file=ops/tpu-cdi/freetpu_cdi.py \
  --from-file=ops/tpu-exporter/tpu_exporter.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

- [manifests/k8s-tpu-cdi-daemonset.yaml](manifests/k8s-tpu-cdi-daemonset.yaml) — Node-level DaemonSets targeting VFIO-backed GKE TPU node pools (`tpu-v5-lite-device`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`): `freetpu-node-agent` runs by default with `--no-pid-scan --tpu-info-bin ""` (no `hostPID`, no `CAP_SYS_PTRACE`, no `Unconfined` AppArmor) to serve **PCI/VFIO hardware inventory (`vfio_group_exclusive`) and VFIO readiness only** on `:9400/metrics` (with isolated loopback liveness/readiness probes on `127.0.0.1:9401`, without publishing CDI specs), plus a `NetworkPolicy` (`freetpu-node-agent-metrics`) restricting `:9400` ingress to the trusted `monitoring` Prometheus scraper. Because GKE Standard accepts `NetworkPolicy` objects even when policy enforcement is disabled, on shared/multi-tenant clusters (including default inventory-only mode) enforced NetworkPolicy plus admission controls forbidding tenant `hostNetwork: true` pods—or binding `--host 127.0.0.1` behind an authenticating proxy such as `kube-rbac-proxy`—is a prerequisite so tenant workloads cannot starve the exporter's bounded `:9400` scrape worker pool. Host PID telemetry (`freetpu_tpu_active_pids`) is an additional explicit opt-in: enable `hostPID: true`, `CAP_SYS_PTRACE`, `appArmorProfile.type: Unconfined` (on GKE), and remove `--no-pid-scan` only after verifying that ingress isolation. A separate opt-in `freetpu-cdi-sync` DaemonSet (`--unlink-on-exit`) publishes boot-scoped `/var/run/cdi/freetpu.json` ONLY on nodes explicitly labeled `freetpu.dev/cdi-enabled=true` after verifying containerd is not an affected CVE-2026-53492 release (upstream affected: `2.1.0–2.1.8`, `2.2.0–2.2.4`, `2.3.0–2.3.1`; fixed in `2.1.9`, `2.2.5`, `2.3.2`, `>= 2.4.0`; upstream `2.0.x` lacks the affected checkpoint-restore path, though vendor/GKE builds should still be verified for backports) or blocking untrusted checkpoint restores. For Kubernetes DRA scheduling, pair the published CDI specs with a cluster DRA driver registered for `freetpu.dev` (and on GKE, coordinate exclusive chip ownership or use a dedicated TPU pool without `google.com/tpu` scheduling). Because CDI device injection does not configure `RLIMIT_MEMLOCK` or capabilities and Kubernetes Pods have no `--ulimit` field, also verify each TPU workload's effective memlock limit before running VFIO workloads—on self-managed nodes inheriting a restrictive systemd memlock limit, raise the container runtime's memlock limit (e.g. `LimitMEMLOCK=infinity` in `containerd.service` or containerd's default OCI spec) or grant an approved `IPC_LOCK` capability to TPU workload containers.
