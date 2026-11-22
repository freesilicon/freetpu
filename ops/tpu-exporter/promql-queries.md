# FreeTPU Prometheus Billing & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for VFIO-backed Cloud TPU (`v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py`. (Legacy `/dev/accel*` Cloud TPU v4 hosts are recognized in `freetpu_tpu_info` PCI inventory only; VFIO health and `/dev/vfio/*` PID metrics apply to VFIO-backed hosts.)

## Fleet Inventory & Health

> **Deduplication note:** `prometheus-scrape.yml` attaches a stable `node` label (and `collector_role="node"|"workload-sidecar"`) to every scraped target. The fleet inventory, health, coverage, and spend queries below deduplicate by `(generation, node, pci_bdf)` so running both the node DaemonSet and one or more workload telemetry sidecars on the same node counts every physical TPU chip **exactly once**.

```promql
# Total discovered physical TPU chips per generation across scraped nodes
count by (generation) (
  group by (generation, node, pci_bdf) (freetpu_tpu_info)
)

# Total VFIO-ready physical TPU chips per generation across the cluster
count by (generation) (
  min by (generation, node, pci_bdf) (freetpu_tpu_health_status) == 1
)

# Unhealthy or unbound TPU chips (vfio-pci / /dev/vfio/<group> missing)
min by (generation, node, pci_bdf) (freetpu_tpu_health_status) == 0
```

## Compute & HBM Utilization

```promql
# Per-chip MXU / TensorCore duty cycle (%) deduplicated by physical chip
max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent)

# Average cluster MXU utilization by TPU generation
avg by (generation) (max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent))

# HBM allocation ratio (0.0 - 1.0) per physical chip
max by (generation, node, pci_bdf) (freetpu_tpu_hbm_utilization_ratio)

# HBM used in GiB per physical chip
max by (generation, node, pci_bdf) (freetpu_tpu_hbm_used_bytes) / (1024 * 1024 * 1024)
```

## Billing, Chargeback & Idle Waste Detection

```promql
# Active chip-hours accrued over 24h per physical chip (generation, node, pci_bdf):
# chip has >= 1 active host PID (from the node collector) OR duty cycle > 1%
# (from a workload sidecar or local collector). Both signals are aggregated to
# the same (generation, node, pci_bdf) label set before `or` so one physical
# chip never accrues hours across multiple collector series.
sum_over_time(
  (
    clamp_max(max by (generation, node, pci_bdf) (freetpu_tpu_active_pids{collector_role="node"}) > 0, 1)
    or
    ((max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent) > 1) * 0 + 1)
  )[24h:1m]
) / 60

# Idle reserved chips per physical chip (generation, node, pci_bdf):
# host-wide PID reservation (collector_role="node") persisted for the full 15m
# window AND sufficient samples exist across the 15m window AND average MXU
# duty cycle (from a workload sidecar or local collector) < 2%.
# Aggregating each signal by (generation, node, pci_bdf) joins node-level PID
# evidence with sidecar duty-cycle evidence cleanly across distinct `instance`
# and `collector_role` labels.
# Assumes the 15s scrape_interval from prometheus-scrape.yml (60 samples / 15m;
# requiring >= 55 samples avoids flagging newly started workloads early. Scale
# the threshold proportionally for other scrape intervals, e.g. >= 27 for 30s).
(
  min by (generation, node, pci_bdf) (
    min_over_time(freetpu_tpu_active_pids{collector_role="node"}[15m])
  ) > 0
)
  and
(
  max by (generation, node, pci_bdf) (
    count_over_time(freetpu_tpu_active_pids{collector_role="node"}[15m])
  ) >= 55
)
  and
(
  max by (generation, node, pci_bdf) (
    count_over_time(freetpu_tpu_duty_cycle_percent[15m])
  ) >= 55
)
  and
(
  max by (generation, node, pci_bdf) (
    avg_over_time(freetpu_tpu_duty_cycle_percent[15m])
  ) < 2
)

# Duty-cycle telemetry coverage ratio per generation (0.0 - 1.0).
# Deduplicates physical chips by (generation, node, pci_bdf) across node
# DaemonSet and workload sidecar targets. Check coverage before extrapolating
# utilization-adjusted costs across the fleet.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent))
  /
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info))
)

# Inventory-based estimated hourly TPU spend by generation across scraped nodes.
# Counts each physical PCI chip ONCE via `group by (generation, node, pci_bdf)`
# on `freetpu_tpu_info` (never double-counting when a node runs both the
# DaemonSet and workload sidecars, and still counting unhealthy/unbound chips
# that Google Cloud bills while the TPU node is READY).
# Replace 1.20 / 2.70 with your spot/on-demand SKU rate; use Google Cloud billing
# or billable node inventory for authoritative chargeback.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{generation="v6e"})) * 2.70
)

# Measured-chip effective cost per utilized TPU-hour by generation.
# Restricts BOTH numerator and denominator to deduplicated physical chips
# `(generation, node, pci_bdf)` with usable, PCI-mapped
# `freetpu_tpu_duty_cycle_percent` readings so neither partial telemetry coverage
# nor multiple collectors per node skew the estimate. Check the telemetry
# coverage ratio above before extrapolating fleet-wide.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{generation="v6e"})) * 2.70
) / clamp_min(sum by (generation) (max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent) / 100), 0.01)
```
