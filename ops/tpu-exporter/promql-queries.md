# FreeTPU Prometheus Billing & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for VFIO-backed Cloud TPU (`v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py`. (Legacy `/dev/accel*` Cloud TPU v4 hosts are recognized in `freetpu_tpu_info` PCI inventory only; VFIO health and `/dev/vfio/*` PID metrics apply to VFIO-backed hosts.)

## Fleet Inventory & Health

> **Trust & Prometheus isolation note:** FreeTPU separates scraping across two Prometheus configurations:
> 1. **`prometheus-scrape.yml` (Trusted Billing & Node Prometheus):** Scrapes ONLY operator-controlled `collector_role="node"` targets (standalone VMs and the `kube-system` `freetpu-node-agent` DaemonSet). All fleet inventory, health, reservation, and billing/chargeback queries run on this instance and deduplicate by `(generation, node, pci_bdf)` so tenant pods cannot affect billing metrics or storage.
> 2. **`prometheus-scrape-diagnostics.yml` (Isolated Diagnostic Prometheus):** Scrapes `freetpu-workload-sidecar` pods in `ml-workloads`, strips `pci_bdf`, and constrains `chip`/`runtime_chip` to `0-63` (covering v4/v5e/v5p/v6e runtime device IDs while bounding cardinality) so tenant workloads cannot create unbounded series churn. Sidecar utilization queries below run on this separate diagnostic instance (grouping by `(generation, node, chip)` or `runtime_chip`).

```promql
# Total discovered physical TPU chips per generation across scraped nodes
count by (generation) (
  group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
)

# Total VFIO-ready physical TPU chips per generation across the cluster
count by (generation) (
  min by (generation, node, pci_bdf) (freetpu_tpu_health_status{collector_role="node"}) == 1
)

# Unhealthy or unbound TPU chips (vfio-pci / /dev/vfio/<group> missing)
min by (generation, node, pci_bdf) (freetpu_tpu_health_status{collector_role="node"}) == 0
```

## Compute & HBM Utilization

### Runtime-Keyed Utilization (Works Out-of-the-Box with Default Sidecar)

The default workload sidecar (`k8s-tpu-workload-telemetry-sidecar.yaml`) runs without `--chip-id-map` and publishes live `libtpu` readings under `freetpu_tpu_runtime_*`:

```promql
# Per-runtime-chip MXU / TensorCore duty cycle (%) — works out of the box
freetpu_tpu_runtime_duty_cycle_percent

# Average runtime MXU utilization (%) per node
avg by (node) (freetpu_tpu_runtime_duty_cycle_percent)

# HBM allocation ratio (0.0 - 1.0) per runtime chip
freetpu_tpu_runtime_hbm_used_bytes / (freetpu_tpu_runtime_hbm_total_bytes > 0)

# HBM used in GiB per runtime chip
freetpu_tpu_runtime_hbm_used_bytes / (1024 * 1024 * 1024)
```

### Per-Chip Utilization (Diagnostic Prometheus or Single-Tenant VM)

On the isolated diagnostic Prometheus instance (`prometheus-scrape-diagnostics.yml`), `pci_bdf` is stripped from sidecar series to prevent tenant cardinality churn, so per-chip diagnostic queries group by `(generation, node, chip)` (requires a verified `--chip-id-map` or sysfs `chip_id` on the collector). Sidecar-reported utilization is tenant-supplied and must be used for **diagnostics only**, not financial chargeback:

```promql
# Per-chip MXU / TensorCore duty cycle (%) on the diagnostic Prometheus instance
max by (generation, node, chip) (freetpu_tpu_duty_cycle_percent)

# Average cluster MXU utilization by TPU generation
avg by (generation) (max by (generation, node, chip) (freetpu_tpu_duty_cycle_percent))

# HBM allocation ratio (0.0 - 1.0) per chip
max by (generation, node, chip) (freetpu_tpu_hbm_utilization_ratio)

# HBM used in GiB per chip
max by (generation, node, chip) (freetpu_tpu_hbm_used_bytes) / (1024 * 1024 * 1024)
```

## Billing & Chargeback (Operator-Controlled `collector_role="node"` Only)

> **Trust boundary:** Every financial, reservation-hour, and estimated-spend query in this section reads **exclusively** from operator-controlled `collector_role="node"` series (`freetpu_tpu_info` and `freetpu_tpu_active_pids` from the trusted `freetpu-node-exporter` DaemonSet or standalone host VM scrape). Tenant-supplied workload sidecar metrics (`collector_role="workload-sidecar"`) are **excluded** because a pod on a TPU node controls its HTTP scrape payload and can report arbitrary `generation`, `pci_bdf`, or utilization values for chips on that node. These queries do not require `--chip-id-map`.

```promql
# 24h PID telemetry coverage ratio per physical chip (0.0 - 1.0) at the 15s billing
# scrape cadence (`5760 = 24h / 15s` scrapes). Using `max_over_time(...[15s])`
# inside `[24h:15s]` requires a fresh sample per 15s interval (preventing
# Prometheus's default 5m lookback delta from masking missed scrapes).
# Falling back to `max_over_time(freetpu_tpu_info{collector_role="node"}[24h])`
# keeps chips visible as 0 coverage even if the exporter goes down and its instant
# inventory series becomes stale (pair with independent cloud billable node
# inventory to detect outages lasting longer than 24h).
(
  count_over_time(
    (
      max by (generation, node, pci_bdf) (
        max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      )
    )[24h:15s]
  )
  or
  (0 * max by (generation, node, pci_bdf) (max_over_time(freetpu_tpu_info{collector_role="node"}[24h])))
) / 5760

# Audit / alert: chips with ANY incomplete 24h PID telemetry (< 100% of 5760 15s
# scrapes, including full exporter outages observed during the 24h window).
# Any chip returned here is excluded from automatic chargeback totals below and
# must be reconciled before billing.
(
  (
    count_over_time(
      (
        max by (generation, node, pci_bdf) (
          max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
        )
      )[24h:15s]
    )
    or
    (0 * max by (generation, node, pci_bdf) (max_over_time(freetpu_tpu_info{collector_role="node"}[24h])))
  ) / 5760
) < 1

# Missing 15s telemetry intervals over 24h per physical chip (0 = complete window).
5760 - (
  count_over_time(
    (
      max by (generation, node, pci_bdf) (
        max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      )
    )[24h:15s]
  )
  or
  (0 * max by (generation, node, pci_bdf) (max_over_time(freetpu_tpu_info{collector_role="node"}[24h])))
)

# Operator-observed active chip-hours over 24h per physical chip (generation, node, pci_bdf):
# Evaluates fresh samples at the 15s billing scrape cadence (`240 = 1h / 15s`
# scrapes per hour, `5760 = 24h / 15s` scrapes per day) so sub-minute bursts
# captured by the 15s scraper are counted accurately.
# Uses ONLY operator-controlled host VFIO PID handles (`collector_role="node"`)
# for chips whose full `/sys/kernel/iommu_groups/<group>/devices` membership is
# verified exclusive to that single TPU (`vfio_group_exclusive="true"`), AND
# requires COMPLETE 24h PID telemetry coverage (`== 1`, all 5760/5760 scrapes)
# so missing scans/scrapes never silently understate a chargeback total.
sum_over_time(
  (
    max by (generation, node, pci_bdf) (
      max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (
        freetpu_tpu_info{collector_role="node", vfio_group_exclusive="true"}
      )
    ) > bool 0
  )[24h:15s]
) / 240
and on (generation, node, pci_bdf)
(
  count_over_time(
    (
      max by (generation, node, pci_bdf) (
        max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      )
    )[24h:15s]
  ) / 5760 == 1
)

# Shared or non-exclusive VFIO-group reservation hours over 24h
# (`collector_role="node"` only, evaluated at the 15s scrape cadence; routes
# groups with `vfio_group_exclusive="false"` — including groups shared with
# sibling TPUs, non-TPU PCI devices, or unverified IOMMU group membership — to
# group-level reservation hours rather than per-chip chargeback). Requires
# COMPLETE 24h PID telemetry coverage (`== 1`, all 5760/5760 scrapes); incomplete
# windows are suppressed here and surfaced by the coverage audit above.
sum_over_time(
  (
    max by (generation, node, vfio_device) (
      max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (
        freetpu_tpu_info{collector_role="node", vfio_group_exclusive="false"}
      )
    ) > bool 0
  )[24h:15s]
) / 240
and on (generation, node, vfio_device)
(
  count_over_time(
    (
      max by (generation, node, vfio_device) (
        max_over_time(freetpu_tpu_active_pids{collector_role="node"}[15s])
      )
    )[24h:15s]
  ) / 5760 == 1
)

# Inventory-based estimated hourly TPU spend by generation across scraped nodes.
# Counts each trusted physical PCI chip ONCE via `collector_role="node"` and
# `group by (generation, node, pci_bdf)` on `freetpu_tpu_info` (never trusting
# tenant sidecar inventory, never double-counting, and still counting
# unhealthy/unbound chips that Google Cloud bills while the TPU node is READY).
# Includes PF-only Trillium nodes (`generation="v6e-pf"`, PCI 0x006e) at the
# Trillium rate alongside VF Trillium nodes (`generation="v6e"`, PCI 0x006f).
# Replace the placeholder per-chip hourly SKU rates (v5e: 1.20, v5p: 4.20,
# v6e/v6e-pf: 2.70) with your contract/spot/on-demand rates; use Google Cloud
# billing or billable node inventory for authoritative chargeback.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5p"})) * 4.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation=~"v6e|v6e-pf"})) * 2.70
)

# Operator-controlled effective hourly cost per reserved/active exclusive chip by generation.
# Uses ONLY trusted `collector_role="node"` inventory and host VFIO PID reservation
# evidence (`freetpu_tpu_active_pids{collector_role="node"} > 0`) on verified
# exclusive `/dev/vfio/<group>` chips (`vfio_group_exclusive="true"`). Omits
# generations with zero active exclusive chips.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation="v5p"})) * 4.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node", generation=~"v6e|v6e-pf"})) * 2.70
) / (
  count by (generation) (
    max by (generation, node, pci_bdf) (
      freetpu_tpu_active_pids{collector_role="node"}
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (
        freetpu_tpu_info{collector_role="node", vfio_group_exclusive="true"}
      )
    ) > 0
  ) > 0
)
```

## Workload Utilization Diagnostics & Idle Waste Detection (Diagnostic-Only / Non-Billing)

> **Non-authoritative & deployment note:**
> - **Diagnostic Kubernetes sidecars (`prometheus-scrape-diagnostics.yml`):** Run on a **separate diagnostic Prometheus instance** where `pci_bdf` is stripped and only low-cardinality `runtime_chip`/`chip` (`0-63`) labels are kept. Use the sidecar-native idle check below on that diagnostic instance.
> - **Operator-controlled single-tenant hosts (`collector_role="node"`):** The `pci_bdf`-joined idle, coverage, and efficiency queries below assume both `freetpu_tpu_info` and `freetpu_tpu_duty_cycle_percent` are scraped from trusted operator-controlled `collector_role="node"` targets (single-tenant Cloud TPU VMs running `tpu-info` locally with verified `--chip-id-map` or sysfs `chip_id`). Never mix untrusted tenant sidecar scrapes into the billing Prometheus instance.

```promql
# Diagnostic (isolated diagnostic Prometheus, `prometheus-scrape-diagnostics.yml`):
# Workload sidecar reporting sustained low MXU duty cycle (< 2%) across 15m per
# runtime chip (requires >= 55 samples at 15s scrape_interval).
(
  max by (node, namespace, pod, runtime_chip) (
    count_over_time(freetpu_tpu_runtime_duty_cycle_percent{collector_role="workload-sidecar"}[15m])
  ) >= 55
)
  and
(
  max by (node, namespace, pod, runtime_chip) (
    avg_over_time(freetpu_tpu_runtime_duty_cycle_percent{collector_role="workload-sidecar"}[15m])
  ) < 2
)

# Operator-controlled single-tenant VM diagnostic: Idle reserved chips per
# physical chip (generation, node, pci_bdf) where host PID reservation persisted
# for 15m AND operator-collected duty cycle < 2%.
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
    count_over_time(freetpu_tpu_duty_cycle_percent{collector_role="node"}[15m])
  ) >= 55
)
  and
(
  max by (generation, node, pci_bdf) (
    avg_over_time(freetpu_tpu_duty_cycle_percent{collector_role="node"}[15m])
  ) < 2
)

# Operator-controlled single-tenant VM diagnostic: Duty-cycle telemetry coverage
# ratio per generation (0.0 - 1.0) across `collector_role="node"` collectors.
(
  (
    count by (generation) (
      group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{collector_role="node"})
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
    )
    or
    (0 * count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})))
  )
  /
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"}))
)

# Operator-controlled single-tenant VM diagnostic (non-billing): Estimated cost
# per utilized TPU-hour by generation (`collector_role="node"` only). Do NOT use
# tenant sidecar readings for financial chargeback.
(
  count by (generation) (
    group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{collector_role="node", generation="v5e"})
    and on (generation, node, pci_bdf)
    group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
  ) * 1.20
  or
  count by (generation) (
    group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{collector_role="node", generation="v5p"})
    and on (generation, node, pci_bdf)
    group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
  ) * 4.20
  or
  count by (generation) (
    group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{collector_role="node", generation=~"v6e|v6e-pf"})
    and on (generation, node, pci_bdf)
    group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
  ) * 2.70
) / (
  sum by (generation) (
    (
      max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{collector_role="node"})
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
    ) / 100
  ) > 0
)
```
