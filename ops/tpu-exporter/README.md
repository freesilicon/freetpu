# `tpu-exporter` — Prometheus Telemetry for Cloud TPUs

`tpu_exporter.py` is a zero-dependency Python 3 Prometheus exporter (analogous to NVIDIA `dcgm-exporter`) for Google Cloud TPU VMs (`v4`, `v5e`, `v5p`, `v6e` Trillium).

## What It Discovers & Exports

1. **PCI & VFIO Topology (`/sys/bus/pci/devices/*` + `/dev/vfio/*`)**
   - Filters Google PCI vendor `0x1ae0` and maps known TPU device IDs (`0x0027`, `0x0056`, `0x005e`, `0x0062`, `0x0063`, `0x006e`, `0x006f`).
   - Resolves IOMMU group, bound kernel driver (`vfio-pci`), `/dev/vfio/<group>` node, and `/dev/vfio/vfio` container readiness.
2. **Runtime Telemetry (`tpu-info` + `/proc/*/fd` fallback)**
   - Scrapes HBM used/total bytes, MXU duty cycle `%`, active PIDs, and chip health status (`freetpu_tpu_health_status`).
   - Falls back cleanly to sysfs + `/proc/<pid>/fd` VFIO handle scanning when `tpu-info` is unavailable.

## Usage

```bash
# HTTP /metrics server on localhost:9400 (default; pass --host 0.0.0.0 explicitly in containers)
python3 ops/tpu-exporter/tpu_exporter.py --port 9400

# One-shot Prometheus text exposition snapshot (CI / ephemeral Colab runs)
python3 ops/tpu-exporter/tpu_exporter.py --once

# Structured JSON snapshot
python3 ops/tpu-exporter/tpu_exporter.py --once --json
```

See [prometheus-scrape.yml](prometheus-scrape.yml) for Prometheus scrape configuration and [promql-queries.md](promql-queries.md) for billing and utilization PromQL queries.
