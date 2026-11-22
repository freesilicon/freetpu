# FreeTPU Prometheus Billing & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for VFIO-backed Cloud TPU (`v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py`. (Legacy `/dev/accel*` Cloud TPU v4 hosts are recognized in `freetpu_tpu_info` PCI inventory only; VFIO health and `/dev/vfio/*` PID metrics apply to VFIO-backed hosts.)

## Fleet Inventory & Health

```promql
# Total discovered TPU chips per generation across scraped hosts (deduplicated per chip)
count by (generation) (
  group by (generation, instance, pci_bdf) (freetpu_tpu_info)
)

# Total VFIO-ready TPU chips per generation across the cluster
sum by (generation) (freetpu_tpu_health_status == 1)

# Unhealthy or unbound TPU chips (vfio-pci / /dev/vfio/<group> missing)
freetpu_tpu_health_status == 0
```

## Compute & HBM Utilization

```promql
# Per-chip MXU / TensorCore duty cycle (%)
freetpu_tpu_duty_cycle_percent

# Average cluster MXU utilization by TPU generation
avg by (generation) (freetpu_tpu_duty_cycle_percent)

# HBM allocation ratio (0.0 - 1.0) per chip
freetpu_tpu_hbm_utilization_ratio

# HBM used in GiB per chip
freetpu_tpu_hbm_used_bytes / (1024 * 1024 * 1024)
```

## Billing, Chargeback & Idle Waste Detection

```promql
# Active chip-hours accrued over 24h (chip has >= 1 active PID or duty cycle > 1%)
sum_over_time(
  (
    clamp_max(freetpu_tpu_active_pids > 0, 1)
    or
    ((freetpu_tpu_duty_cycle_percent > 1) * 0 + 1)
  )[24h:1m]
) / 60

# Idle reserved chips: PID reservation persisted for the full 15m window AND
# sufficient samples exist across the 15m window AND average MXU duty cycle < 2%.
# Assumes the 15s scrape_interval from prometheus-scrape.yml (60 samples / 15m;
# requiring >= 55 samples avoids flagging newly started workloads early. Scale
# the threshold proportionally for other scrape intervals, e.g. >= 27 for 30s).
(min_over_time(freetpu_tpu_active_pids[15m]) > 0)
  and
(count_over_time(freetpu_tpu_active_pids[15m]) >= 55)
  and
(count_over_time(freetpu_tpu_duty_cycle_percent[15m]) >= 55)
  and
(avg_over_time(freetpu_tpu_duty_cycle_percent[15m]) < 2)

# Duty-cycle telemetry coverage ratio per generation (0.0 - 1.0).
# Check coverage before extrapolating utilization-adjusted costs across the fleet.
(
  count by (generation) (freetpu_tpu_duty_cycle_percent)
  /
  count by (generation) (group by (generation, instance, pci_bdf) (freetpu_tpu_info))
)

# Inventory-based estimated hourly TPU spend by generation across scraped hosts.
# Counts all discovered PCI chips via `freetpu_tpu_info` (including unhealthy or
# unbound chips that Google Cloud still bills while the TPU node is READY),
# deduplicated per (generation, instance, pci_bdf) against label churn.
# Replace 1.20 / 2.70 with your spot/on-demand SKU rate; use Google Cloud billing
# or billable node inventory for authoritative chargeback.
(
  count by (generation) (group by (generation, instance, pci_bdf) (freetpu_tpu_info{generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, instance, pci_bdf) (freetpu_tpu_info{generation="v6e"})) * 2.70
)

# Measured-chip effective cost per utilized TPU-hour by generation.
# Restricts BOTH numerator and denominator to chips with usable, PCI-mapped
# `freetpu_tpu_duty_cycle_percent` readings so partial telemetry coverage never
# inflates the estimate. Check the telemetry coverage ratio above before
# extrapolating fleet-wide.
(
  count by (generation) (freetpu_tpu_duty_cycle_percent{generation="v5e"}) * 1.20
  or
  count by (generation) (freetpu_tpu_duty_cycle_percent{generation="v6e"}) * 2.70
) / clamp_min(sum by (generation) (freetpu_tpu_duty_cycle_percent / 100), 0.01)
```
