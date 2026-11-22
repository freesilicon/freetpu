# `freetpu-cdi` — Open CNCF CDI Toolkit for Cloud TPUs

`freetpu_cdi.py` is an open, zero-dependency CNCF Container Device Interface (CDI v0.6.0) spec generator for Google Cloud TPUs (analogous to `nvidia-ctk cdi generate`). It scans `/sys/bus/pci/devices/*` (`vendor=0x1ae0`) and `/dev/vfio/*`, and emits `/etc/cdi/freetpu.json` so Docker, Podman, containerd, and Kubernetes can attach TPU chips cleanly without privileged host-network/host-device hacks.

## Quickstart

```bash
# 1. Generate and validate /etc/cdi/freetpu.json on a Cloud TPU VM
sudo python3 ops/tpu-cdi/freetpu_cdi.py generate --output /etc/cdi/freetpu.json --validate

# 2. Preview to stdout without writing /etc/cdi
python3 ops/tpu-cdi/freetpu_cdi.py generate --stdout --validate

# 3. Validate an existing CDI spec
python3 ops/tpu-cdi/freetpu_cdi.py validate /etc/cdi/freetpu.json
```

## Docker / Podman / nerdctl Passthrough

With `/etc/cdi/freetpu.json` generated (and CDI enabled in Docker/containerd):

```bash
# Attach all TPU chips on the host with VFIO memlock + CAP_IPC_LOCK
docker run --rm \
  --device=freetpu.dev/tpu=all \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  python:3.12-slim \
  python3 -c "import os; print(os.listdir('/dev/vfio'))"

# Attach only chip 0
docker run --rm \
  --device=freetpu.dev/tpu=0 \
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

Note on shared IOMMU groups: if multiple TPU chips share one `/dev/vfio/<iommu_group>` node, `freetpu-cdi` exposes that group strictly as a single allocation unit (`freetpu.dev/tpu=group-<iommu_group>`) plus `freetpu.dev/tpu=all` instead of misleading single-chip selectors.

## Kubernetes Manifests

First create the `freetpu-ops-scripts` ConfigMap in `kube-system`:

```bash
kubectl -n kube-system create configmap freetpu-ops-scripts \
  --from-file=ops/tpu-cdi/freetpu_cdi.py \
  --from-file=ops/tpu-exporter/tpu_exporter.py \
  --dry-run=client -o yaml | kubectl apply -f -
```

- [manifests/k8s-tpu-cdi-daemonset.yaml](manifests/k8s-tpu-cdi-daemonset.yaml) — Node-level DaemonSet generating `/var/run/cdi/freetpu.json` and serving Prometheus metrics on `:9400/metrics` (with `SYS_PTRACE` for hostPID `/proc/<pid>/fd` telemetry).
- [manifests/k8s-tpu-pod-cdi.yaml](manifests/k8s-tpu-pod-cdi.yaml) — Sample Kubernetes `DeviceClass`, DRA `ResourceClaimTemplate`, and Pod requesting `freetpu.dev/tpu` (requires Kubernetes >= 1.32 with DRA enabled and a cluster DRA driver registered for `freetpu.dev` publishing `ResourceSlice` objects).
