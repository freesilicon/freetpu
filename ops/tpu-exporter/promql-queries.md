# FreeTPU Prometheus Fleet & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for VFIO-backed Cloud TPU (`v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py` and scraped via `ops/tpu-exporter/prometheus-scrape.yml`. (Legacy `/dev/accel*` Cloud TPU v4 hosts are recognized in `freetpu_tpu_info` PCI inventory only; VFIO health and `/dev/vfio/*` PID metrics apply to VFIO-backed hosts.)

## Fleet Inventory & Health

All fleet inventory and VFIO health queries read exclusively from operator-controlled `collector_role="node"` targets (standalone Cloud TPU VMs and the `kube-system` `freetpu-node-agent` DaemonSet) and deduplicate by `(cluster, generation, node, pci_bdf)` so multi-cluster scrapes with overlapping node names never merge distinct chips.

```promql
# Total discovered physical TPU chips per generation across scraped clusters/nodes
count by (generation) (
  group by (cluster, generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
)

# Total VFIO-ready physical TPU chips per generation across the cluster (VFIO-backed generations;
# preserves 0 when all discovered chips in a generation are unhealthy)
sum by (generation) (
  min by (cluster, generation, node, pci_bdf) (
    freetpu_tpu_health_status{collector_role="node", generation=~"v5e|v5p|v6e|v6e-pf"}
  ) == bool 1
)

# Unhealthy or unbound VFIO-backed TPU chips (vfio-pci / /dev/vfio/<group> missing or unverified;
# excludes legacy /dev/accel* v2/v3/v4 hosts where VFIO readiness is inapplicable)
min by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_health_status{collector_role="node", generation=~"v5e|v5p|v6e|v6e-pf"}
) == 0
```

## Compute & HBM Utilization (Hosts & Operator-Managed Workload Collectors Running `tpu-info`)

Live HBM and MXU duty-cycle metrics come from either:
- Standalone Cloud TPU VMs running `tpu-info` (`collector_role="node"`), or
- Operator-managed workload-namespace collectors running `tpu_exporter.py --no-pid-scan --runtime-metrics-addr 127.0.0.1:8431` scraped under a dedicated `collector_role="workload"` label (kept separate from `collector_role="node"` so workload-collector inventory/VFIO samples never mix into node fleet queries).

```promql
# Per-runtime-chip MXU / TensorCore duty cycle (%) — works without --chip-id-map
freetpu_tpu_runtime_duty_cycle_percent{collector_role=~"node|workload"}

# Average runtime MXU utilization (%) per cluster and node
avg by (cluster, node) (freetpu_tpu_runtime_duty_cycle_percent{collector_role=~"node|workload"})

# HBM allocation ratio (0.0 - 1.0) per runtime chip
freetpu_tpu_runtime_hbm_used_bytes{collector_role=~"node|workload"}
  / (freetpu_tpu_runtime_hbm_total_bytes{collector_role=~"node|workload"} > 0)

# HBM used in GiB per runtime chip
freetpu_tpu_runtime_hbm_used_bytes{collector_role=~"node|workload"} / (1024 * 1024 * 1024)
```

When `--chip-id-map` or sysfs `chip_id` provides a verified physical chip mapping:

```promql
# Per-physical-chip MXU / TensorCore duty cycle (%)
max by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_duty_cycle_percent{collector_role=~"node|workload"}
)

# Average cluster MXU utilization by cluster and TPU generation
avg by (cluster, generation) (
  max by (cluster, generation, node, pci_bdf) (
    freetpu_tpu_duty_cycle_percent{collector_role=~"node|workload"}
  )
)

# HBM allocation ratio (0.0 - 1.0) per physical chip
max by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_hbm_utilization_ratio{collector_role=~"node|workload"}
)

# HBM used in GiB per physical chip
max by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_hbm_used_bytes{collector_role=~"node|workload"}
) / (1024 * 1024 * 1024)
```

## Host VFIO Process Activity & Inventory Cost Estimates (Observational)

> **Operational vs. Billing Note:**
> - `freetpu_tpu_active_pids` reports processes currently holding an open non-`O_PATH` `/dev/vfio/<group>` group descriptor (`freetpu_tpu_pid_telemetry_available == 1` confirms both `/proc` readability and kernel sysfs VFIO device-number verification). Because Linux allows a process to retain a VFIO device descriptor returned by `VFIO_GROUP_GET_DEVICE_FD` after closing the group descriptor, a zero count is **observational** and must not be treated as proof of an idle chip for automated financial chargeback.
> - Use Google Cloud billing or your cluster's authoritative billable node inventory for financial chargeback; the inventory spend query below provides an instantaneous fleet estimate only.

```promql
# Chips with missing/unverified host PID telemetry (unreadable /proc or unverified /dev/vfio/<group>)
min by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_pid_telemetry_available{collector_role="node"}
) == 0

# Currently observed chips with an open host /dev/vfio/<group> handle
max by (cluster, generation, node, pci_bdf) (
  freetpu_tpu_active_pids{collector_role="node"}
) > 0

# Inventory-based estimated hourly TPU spend by generation across scraped clusters/nodes.
# Counts each physical PCI chip ONCE via `collector_role="node"` and
# `group by (cluster, generation, node, pci_bdf)` on `freetpu_tpu_info` (including
# unhealthy/unbound chips that Google Cloud bills while the TPU node is READY).
# Includes PF-only Trillium nodes (`generation="v6e-pf"`, PCI 0x006e) at the
# Trillium rate alongside VF Trillium nodes (`generation="v6e"`, PCI 0x006f).
# Replace the placeholder per-chip hourly SKU rates (v5e: 1.20, v5p: 4.20,
# v6e/v6e-pf: 2.70) with your contract/spot/on-demand rates.
(
  count by (generation) (group by (cluster, generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5e"})) * 1.20
  or
  count by (generation) (group by (cluster, generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5p"})) * 4.20
  or
  count by (generation) (group by (cluster, generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation=~"v6e|v6e-pf"})) * 2.70
)
```
