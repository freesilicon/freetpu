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

### Per-PCI-Chip Utilization (Requires Verified `--chip-id-map` or Sysfs `chip_id`)

The per-PCI-BDF utilization queries below (and the per-PCI billing, idle-waste, coverage, and cost-per-utilized-TPU-hour queries in the next section) require a verified `--chip-id-map` (or kernel sysfs `chip_id`) binding `libtpu` runtime chip IDs to PCI BDFs:

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

> **Prerequisite:** Queries referencing `freetpu_tpu_duty_cycle_percent` or `freetpu_tpu_hbm_used_bytes` require a verified `--chip-id-map` (or sysfs `chip_id`) on the telemetry collector so runtime readings attach to physical PCI BDFs. Inventory-only spend (`freetpu_tpu_info`) and host PID queries (`freetpu_tpu_active_pids`) do not require `--chip-id-map`.

```promql
# Per-chip active chip-hours accrued over 24h per physical chip (generation, node, pci_bdf):
# Uses chip-specific runtime telemetry (duty cycle > 1% or allocated HBM > 0) for all
# chips, and includes host-wide VFIO PID handles (`collector_role="node"`) ONLY for
# chips with an exclusive (non-shared) `/dev/vfio/<group>` node so a workload opening
# a shared VFIO group never accrues per-chip active hours for idle sibling chips.
sum_over_time(
  (
    clamp_max(
      max by (generation, node, pci_bdf) (
        freetpu_tpu_active_pids{collector_role="node"}
        and on (node, vfio_device)
        (count by (node, vfio_device) (group by (node, vfio_device, pci_bdf) (freetpu_tpu_info)) == 1)
      ) > 0,
      1
    )
    or
    ((max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent) > 1) * 0 + 1)
    or
    ((max by (generation, node, pci_bdf) (freetpu_tpu_hbm_used_bytes) > 0) * 0 + 1)
  )[24h:1m]
) / 60

# Shared-VFIO-group reservation hours over 24h (tracked at group granularity
# `(generation, node, vfio_device)` as reservation evidence rather than per-BDF activity)
sum_over_time(
  clamp_max(
    max by (generation, node, vfio_device) (
      freetpu_tpu_active_pids{collector_role="node"}
      and on (node, vfio_device)
      (count by (node, vfio_device) (group by (node, vfio_device, pci_bdf) (freetpu_tpu_info)) > 1)
    ) > 0,
    1
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
# DaemonSet and workload sidecar targets, and supplies a `0` numerator for any
# discovered generation with zero PCI-mapped duty-cycle readings so low-coverage
# alerts still fire when coverage is 0.
(
  (
    count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent))
    or
    (0 * count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info)))
  )
  /
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info))
)

# Inventory-based estimated hourly TPU spend by generation across scraped nodes.
# Counts each physical PCI chip ONCE via `group by (generation, node, pci_bdf)`
# on `freetpu_tpu_info` (never double-counting when a node runs both the
# DaemonSet and workload sidecars, and still counting unhealthy/unbound chips
# that Google Cloud bills while the TPU node is READY).
# Replace the placeholder per-chip hourly SKU rates (v5e: 1.20, v5p: 4.20,
# v6e: 2.70) with your contract/spot/on-demand rates; use Google Cloud billing
# or billable node inventory for authoritative chargeback.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{generation="v5p"})) * 4.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{generation="v6e"})) * 2.70
)

# Measured-chip effective cost per utilized TPU-hour by generation.
# Restricts BOTH numerator and denominator to deduplicated physical chips
# `(generation, node, pci_bdf)` with usable, PCI-mapped
# `freetpu_tpu_duty_cycle_percent` readings, and filters strictly to positive
# total utilization (`> 0`) so zero-utilization generations are omitted rather
# than reporting a synthetic finite cost and low positive utilizations divide by
# their true measured value. Replace the placeholder SKU rates (v5e: 1.20,
# v5p: 4.20, v6e: 2.70) with your rates, and check the telemetry coverage ratio
# above before extrapolating fleet-wide.
(
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{generation="v5e"})) * 1.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{generation="v5p"})) * 4.20
  or
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent{generation="v6e"})) * 2.70
) / (sum by (generation) (max by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent) / 100) > 0)
```
