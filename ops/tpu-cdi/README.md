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

To inject `CAP_IPC_LOCK` and unlimited `RLIMIT_MEMLOCK` automatically via an OCI `createContainer` hook (no `--cap-add` / `--ulimit` flags required on `docker run`):

```bash
sudo install -m 0755 ops/tpu-cdi/freetpu_cdi.py /usr/local/bin/freetpu-cdi
sudo /usr/local/bin/freetpu-cdi generate \
  --output /etc/cdi/freetpu.json \
  --with-oci-hook /usr/local/bin/freetpu-cdi \
  --validate
```

## Kubernetes Manifests

- [manifests/k8s-tpu-cdi-daemonset.yaml](manifests/k8s-tpu-cdi-daemonset.yaml) — Node-level DaemonSet generating `/var/run/cdi/freetpu.json` and serving Prometheus metrics on `:9400/metrics`.
- [manifests/k8s-tpu-pod-cdi.yaml](manifests/k8s-tpu-pod-cdi.yaml) — Sample Kubernetes Pod & DRA `ResourceClaimTemplate` requesting `freetpu.dev/tpu=all` with `IPC_LOCK`.
