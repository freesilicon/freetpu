# FreeTPU Prometheus Billing & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for Cloud TPU (`v4`, `v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py`.

## Fleet Inventory & Health

```promql
# Total healthy TPU chips per generation across the cluster
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
  (clamp_max(freetpu_tpu_active_pids, 1) or (freetpu_tpu_duty_cycle_percent > bool 1))[24h:1m]
) / 60

# Idle reserved chips (active PID holding /dev/vfio/* but MXU duty cycle < 2% for 15m)
(freetpu_tpu_active_pids > 0)
  and
(avg_over_time(freetpu_tpu_duty_cycle_percent[15m]) < 2)

# Estimated hourly TPU spend by generation (replace 1.20 / 2.70 with your spot/on-demand SKU rate)
sum by (generation) (freetpu_tpu_health_status{generation="v5e"}) * 1.20
+
sum by (generation) (freetpu_tpu_health_status{generation="v6e"}) * 2.70

# Effective cost per utilized TPU-hour (normalized by average MXU duty cycle)
(
  sum by (generation) (freetpu_tpu_health_status)
) / clamp_min(avg by (generation) (freetpu_tpu_duty_cycle_percent) / 100, 0.01)
```
