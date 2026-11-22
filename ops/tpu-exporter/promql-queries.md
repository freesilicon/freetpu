# FreeTPU Prometheus Billing & Utilization PromQL Queries

These queries mirror standard NVIDIA `dcgm-exporter` dashboards for VFIO-backed Cloud TPU (`v5e`, `v5p`, `v6e` Trillium) telemetry exposed by `ops/tpu-exporter/tpu_exporter.py`. (Legacy `/dev/accel*` Cloud TPU v4 hosts are recognized in `freetpu_tpu_info` PCI inventory only; VFIO health and `/dev/vfio/*` PID metrics apply to VFIO-backed hosts.)

## Fleet Inventory & Health

> **Trust & deduplication note:** `prometheus-scrape.yml` splits Kubernetes scraping into a trusted `freetpu-node-exporter` job (scoped by Kubernetes-enforced `kube-system` DaemonSet `freetpu-node-agent` ownership and hard-setting `collector_role="node"`) and a `freetpu-workload-sidecar` job (hard-setting `collector_role="workload-sidecar"`, enforcing response/sample/label limits, and allowlisting only utilization/HBM series). All inventory, health, reservation, and financial/chargeback queries use **strictly** operator-controlled `collector_role="node"` series and deduplicate by `(generation, node, pci_bdf)` so tenant pods cannot fabricate billing inputs and every physical TPU chip is counted **exactly once**. Workload-sidecar utilization queries are kept in a separate diagnostic-only section.

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

### Per-PCI-Chip Utilization (Requires Verified `--chip-id-map` or Sysfs `chip_id`)

The per-PCI-BDF utilization queries below (and the diagnostic idle-waste and coverage queries below) require a verified `--chip-id-map` (or kernel sysfs `chip_id`) binding `libtpu` runtime chip IDs to PCI BDFs. In multi-tenant Kubernetes clusters, sidecar-reported utilization is tenant-supplied and should be used for **diagnostics only**, not financial chargeback:

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

## Billing & Chargeback (Operator-Controlled `collector_role="node"` Only)

> **Trust boundary:** Every financial, reservation-hour, and estimated-spend query in this section reads **exclusively** from operator-controlled `collector_role="node"` series (`freetpu_tpu_info` and `freetpu_tpu_active_pids` from the trusted `freetpu-node-exporter` DaemonSet or standalone host VM scrape). Tenant-supplied workload sidecar metrics (`collector_role="workload-sidecar"`) are **excluded** because a pod on a TPU node controls its HTTP scrape payload and can report arbitrary `generation`, `pci_bdf`, or utilization values for chips on that node. These queries do not require `--chip-id-map`.

```promql
# Operator-observed active chip-hours over 24h per physical chip (generation, node, pci_bdf):
# Uses ONLY operator-controlled host VFIO PID handles (`collector_role="node"`)
# for chips with an exclusive (non-shared) `/dev/vfio/<group>` node so a workload
# opening a shared VFIO group never accrues per-chip hours for idle sibling chips
# and tenant sidecar responses can never alter chargeback hours.
sum_over_time(
  clamp_max(
    max by (generation, node, pci_bdf) (
      freetpu_tpu_active_pids{collector_role="node"}
      and on (node, vfio_device)
      (count by (node, vfio_device) (group by (node, vfio_device, pci_bdf) (freetpu_tpu_info{collector_role="node"})) == 1)
    ) > 0,
    1
  )[24h:1m]
) / 60

# Shared-VFIO-group reservation hours over 24h (operator-controlled `collector_role="node"` only;
# tracked at group granularity `(generation, node, vfio_device)` rather than per-BDF activity)
sum_over_time(
  clamp_max(
    max by (generation, node, vfio_device) (
      freetpu_tpu_active_pids{collector_role="node"}
      and on (node, vfio_device)
      (count by (node, vfio_device) (group by (node, vfio_device, pci_bdf) (freetpu_tpu_info{collector_role="node"})) > 1)
    ) > 0,
    1
  )[24h:1m]
) / 60

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
# evidence (`freetpu_tpu_active_pids{collector_role="node"} > 0`) on exclusive
# `/dev/vfio/<group>` chips. Omits generations with zero active exclusive chips.
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
      and on (node, vfio_device)
      (count by (node, vfio_device) (group by (node, vfio_device, pci_bdf) (freetpu_tpu_info{collector_role="node"})) == 1)
    ) > 0
  ) > 0
)
```

## Workload Utilization Diagnostics & Idle Waste Detection (Diagnostic-Only / Non-Billing)

> **Non-authoritative warning:** The queries below incorporate `freetpu_tpu_duty_cycle_percent` (and optionally `freetpu_tpu_hbm_used_bytes`) scraped from workload sidecars (`collector_role="workload-sidecar"`). In shared Kubernetes clusters, tenant pods control their sidecar scrape response—including `generation`, `pci_bdf`, and utilization values—so these queries are for **engineering diagnostics and advisory idle detection only** and **must not** be used for financial chargeback or billing unless restricted to an operator-controlled collector (`collector_role="node"` on single-tenant VMs with verified `--chip-id-map` or sysfs `chip_id`).

```promql
# Diagnostic: Idle reserved chips per physical chip (generation, node, pci_bdf):
# host-wide PID reservation (collector_role="node") persisted for the full 15m
# window AND sufficient samples exist across the 15m window AND workload-reported
# average MXU duty cycle < 2%.
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

# Diagnostic: Duty-cycle telemetry coverage ratio per generation (0.0 - 1.0).
# Compares PCI-mapped duty-cycle series against trusted `collector_role="node"`
# inventory and supplies a `0` numerator for discovered generations with zero
# PCI-mapped duty-cycle readings. Non-authoritative when scraped from tenant
# workload sidecars (restrict to `collector_role="node"` on operator-owned hosts).
(
  (
    count by (generation) (
      group by (generation, node, pci_bdf) (freetpu_tpu_duty_cycle_percent)
      and on (generation, node, pci_bdf)
      group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})
    )
    or
    (0 * count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"})))
  )
  /
  count by (generation) (group by (generation, node, pci_bdf) (freetpu_tpu_info{collector_role="node"}))
)

# Diagnostic (non-billing): Self-reported efficiency ratio — estimated cost per
# utilized TPU-hour by generation (single-tenant / operator-controlled collectors
# only; add `collector_role="node"` to `freetpu_tpu_duty_cycle_percent` when
# enforcing operator-only inputs). Do NOT use tenant sidecar readings for billing.
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
