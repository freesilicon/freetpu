# `freetpu-cdi` — Open CNCF CDI Toolkit for Cloud TPUs

`freetpu_cdi.py` is an open, zero-dependency CNCF Container Device Interface (CDI v0.6.0) spec generator for VFIO-backed Google Cloud TPUs (`v5e`, `v5p`, `v6e` Trillium bound to `vfio-pci`, analogous to `nvidia-ctk cdi generate`). It scans `/sys/bus/pci/devices/*` (`vendor=0x1ae0`) and `/dev/vfio/*`, verifies full IOMMU group isolation, omits static `major`/`minor` numbers from `deviceNodes` while tracking live char device numbers in `freetpu.dev/vfio-devnums` annotations, and emits a boot-scoped spec at `/var/run/cdi/freetpu.json` (tmpfs, cleared across reboots) so Docker, Podman, containerd (`>= 2.0`), and Kubernetes can attach TPU chips cleanly without privileged host-network/host-device hacks.

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

With boot-scoped `/var/run/cdi/freetpu.json` generated (and CDI enabled in Docker/Podman/containerd `>= 2.0`):

```bash
# Attach all TPU chips on the host with VFIO memlock + CAP_IPC_LOCK
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"

# Attach a single TPU chip by its stable PCI BDF selector (never shifts if another chip disappears)
docker run --rm \
  --device=freetpu.dev/tpu=pci-0000-00-06-0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  my-jax-tpu-image:latest
```

VFIO DMA pinning requires `CAP_IPC_LOCK` and unlimited `RLIMIT_MEMLOCK` configured **before** OCI container startup (`docker run` / `podman run`), because CNCF CDI v0.6.0 does not configure capabilities or rlimits and an OCI `createContainer` hook executes after `runc` loads process settings. Use `python3 ops/tpu-cdi/freetpu_cdi.py docker-args` to emit the required flags:

```bash
docker run --rm \
  $(python3 ops/tpu-cdi/freetpu_cdi.py docker-args --device all) \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"
```

Note on IOMMU groups & stable selectors: single-chip selectors use stable PCI BDF names (`freetpu.dev/tpu=pci-<bdf>`) so a disappearing chip retires its selector instead of reassigning it to a different chip. If multiple TPU chips share one `/dev/vfio/<iommu_group>` node, `freetpu-cdi` exposes that group strictly as a single allocation unit (`freetpu.dev/tpu=group-pci-<bdf1>_<bdf2>`) plus `freetpu.dev/tpu=all`. If any TPU's IOMMU group also contains a non-TPU PCI device, `freetpu-cdi` fails closed rather than publishing any selector (including `all`) that would expose that non-TPU device. Live VFIO char `major:minor` numbers are tracked in `freetpu.dev/vfio-devnums` annotations (while omitted from `deviceNodes`) so `sync` detects unexpected drift and rewrites `/var/run/cdi/freetpu.json` to drop CDI v0.8.0's cached device numbers; before any planned VFIO/PCI unbind or rebind, always stop affected consumers, stop/pause `sync`, delete `/var/run/cdi/freetpu.json` and confirm runtime selector withdrawal, perform the rebind, and then regenerate or resume `sync`.

## Kubernetes Manifests

First create the `freetpu-ops-scripts` ConfigMap in `kube-system`:

```bash
kubectl -n kube-system create configmap freetpu-ops-scripts \
  --from-file=ops/tpu-cdi/freetpu_cdi.py \
  --from-file=ops/tpu-exporter/tpu_exporter.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

- [manifests/k8s-tpu-cdi-daemonset.yaml](manifests/k8s-tpu-cdi-daemonset.yaml) — Node-level DaemonSet targeting VFIO-backed GKE TPU node pools (`tpu-v5-lite-device`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`), reconciling boot-scoped `/var/run/cdi/freetpu.json` via `freetpu-cdi-sync`, and serving **PCI/VFIO inventory + hostPID active-PID metrics only** on `:9400/metrics` (with `SYS_PTRACE` + `appArmorProfile.type: Unconfined` for GKE host `/proc/<pid>/fd` reads) and isolated liveness/readiness probes on `:9401`.
- [manifests/k8s-tpu-workload-telemetry-sidecar.yaml](manifests/k8s-tpu-workload-telemetry-sidecar.yaml) — Workload Pod (deployed in namespace `ml-workloads` to match `ops/tpu-exporter/prometheus-scrape-diagnostics.yml`; create `ml-workloads` first or update both namespace references together) with an unprivileged `tpu-info`-enabled native sidecar sharing only the Pod network namespace (`127.0.0.1:8431`) for live HBM usage and MXU duty-cycle metrics in Kubernetes. **GKE exclusive allocation requirement:** because this sample reserves TPUs through a custom FreeTPU DRA `ResourceClaim` instead of GKE's native `google.com/tpu` extended resource, run it only on a dedicated/tainted TPU node pool where `google.com/tpu` scheduling is unused/disabled, or with a DRA driver that coordinates exclusive chip ownership with GKE's native allocator (otherwise both allocators can assign the same VFIO group concurrently).
- [manifests/k8s-tpu-pod-cdi.yaml](manifests/k8s-tpu-pod-cdi.yaml) — Sample Kubernetes cluster-scoped `DeviceClass`, plus namespaced DRA `ResourceClaimTemplate` and Pod in `ml-workloads` (`resource.k8s.io/v1`, Kubernetes >= 1.34, `containerd >= 2.0`, with `google.com/tpu:NoSchedule` toleration) requesting `freetpu.dev/tpu` (requires a cluster DRA driver registered for `freetpu.dev` publishing `ResourceSlice` objects, plus exclusive TPU ownership on GKE via a dedicated pool or allocator coordination with `google.com/tpu`).
- [manifests/k8s-tpu-pod-cdi-v1beta1.yaml](manifests/k8s-tpu-pod-cdi-v1beta1.yaml) — Legacy Kubernetes 1.32–1.33 DRA sample (`resource.k8s.io/v1beta1`, namespaced resources in `ml-workloads`, `containerd >= 2.0`, requires `--runtime-config=resource.k8s.io/v1beta1=true`, `DynamicResourceAllocation=true`, and the same GKE exclusive-allocation requirement above).
