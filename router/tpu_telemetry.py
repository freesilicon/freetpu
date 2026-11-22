#!/usr/bin/env python3
"""Optional Prometheus `tpu-exporter` scraper and parser for FreeTPU Router.

Parses Prometheus text exposition 0.0.4 metrics emitted by `ops/tpu-exporter`
using the exact metric names from FreeTPU's Pillar 6 TPU exporter:
  - freetpu_exporter_up
  - freetpu_tpu_devices_discovered
  - freetpu_tpu_health_status
  - freetpu_tpu_hbm_used_bytes
  - freetpu_tpu_hbm_total_bytes
  - freetpu_tpu_hbm_utilization_ratio
  - freetpu_tpu_duty_cycle_percent
  - freetpu_tpu_active_pids
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import http.client
import math
import re
import ssl
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

_KNOWN_TPU_METRICS = frozenset(
    {
        "freetpu_exporter_up",
        "freetpu_tpu_devices_discovered",
        "freetpu_tpu_health_status",
        "freetpu_tpu_hbm_used_bytes",
        "freetpu_tpu_hbm_total_bytes",
        "freetpu_tpu_hbm_utilization_ratio",
        "freetpu_tpu_duty_cycle_percent",
        "freetpu_tpu_active_pids",
    }
)

_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?(?:[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|NaN|[+-]?Inf))(?:\s+[0-9]+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass
class PromSample:
    """Single parsed Prometheus metric sample."""

    name: str
    labels: Dict[str, str]
    value: float


@dataclass
class TpuExporterTelemetry:
    """Aggregated TPU node telemetry scraped from a `tpu-exporter` endpoint."""

    reachable: bool
    exporter_up: bool
    devices_discovered: int
    healthy_chips: int
    total_chips: int
    max_hbm_utilization_ratio: Optional[float] = None
    avg_hbm_utilization_ratio: Optional[float] = None
    hbm_used_bytes: Optional[int] = None
    hbm_total_bytes: Optional[int] = None
    avg_duty_cycle_percent: Optional[float] = None
    max_duty_cycle_percent: Optional[float] = None
    active_pids: int = 0
    generations: List[str] = field(default_factory=list)
    scraped_at_unix: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _unescape_label(raw: str) -> str:
    return raw.replace(r"\\", "\\").replace(r"\n", "\n").replace(r"\"", '"')


def parse_prometheus_text(text: str) -> List[PromSample]:
    """Parse Prometheus text exposition format 0.0.4 into `PromSample` items."""
    samples: List[PromSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        name, label_str, val_str = match.group(1), match.group(2), match.group(3)
        try:
            value = float(val_str)
        except ValueError:
            continue
        labels: Dict[str, str] = {}
        if label_str:
            for lmatch in _LABEL_RE.finditer(label_str):
                labels[lmatch.group(1)] = _unescape_label(lmatch.group(2))
        samples.append(PromSample(name=name, labels=labels, value=value))
    return samples


def summarize_tpu_samples(
    samples: List[PromSample],
    *,
    scraped_at: Optional[float] = None,
) -> TpuExporterTelemetry:
    """Aggregate parsed `tpu-exporter` samples into node-level readiness stats."""
    saw_exporter_up = False
    exporter_up = False
    devices_discovered = 0
    health_by_chip: Dict[str, float] = {}
    hbm_used_by_chip: Dict[str, int] = {}
    hbm_total_by_chip: Dict[str, int] = {}
    hbm_ratio_by_chip: Dict[str, float] = {}
    duty_by_chip: Dict[str, float] = {}
    pids_by_chip: Dict[str, int] = {}
    generations_seen: set[str] = set()
    non_finite_metrics: List[str] = []

    for sample in samples:
        if sample.name in _KNOWN_TPU_METRICS and not math.isfinite(sample.value):
            non_finite_metrics.append(sample.name)
            continue

        chip_key = sample.labels.get("chip", sample.labels.get("pci_bdf", "0"))
        gen = sample.labels.get("generation")
        if gen:
            generations_seen.add(gen)

        if sample.name == "freetpu_exporter_up":
            saw_exporter_up = True
            exporter_up = sample.value >= 1.0
        elif sample.name == "freetpu_tpu_devices_discovered":
            devices_discovered = int(round(sample.value))
        elif sample.name == "freetpu_tpu_health_status":
            health_by_chip[chip_key] = sample.value
        elif sample.name == "freetpu_tpu_hbm_used_bytes":
            hbm_used_by_chip[chip_key] = int(round(sample.value))
        elif sample.name == "freetpu_tpu_hbm_total_bytes":
            hbm_total_by_chip[chip_key] = int(round(sample.value))
        elif sample.name == "freetpu_tpu_hbm_utilization_ratio":
            hbm_ratio_by_chip[chip_key] = float(sample.value)
        elif sample.name == "freetpu_tpu_duty_cycle_percent":
            duty_by_chip[chip_key] = float(sample.value)
        elif sample.name == "freetpu_tpu_active_pids":
            pids_by_chip[chip_key] = int(round(sample.value))

    # Derive HBM ratio from used/total if explicit ratio metric was omitted.
    for chip_key, used in hbm_used_by_chip.items():
        if chip_key not in hbm_ratio_by_chip:
            total = hbm_total_by_chip.get(chip_key)
            if total and total > 0:
                ratio = used / float(total)
                if math.isfinite(ratio):
                    hbm_ratio_by_chip[chip_key] = ratio

    total_chips = max(len(health_by_chip), devices_discovered)
    healthy_chips = sum(1 for v in health_by_chip.values() if v >= 1.0)

    ratios = list(hbm_ratio_by_chip.values())
    duties = list(duty_by_chip.values())

    error_msg: Optional[str] = None
    if non_finite_metrics:
        exporter_up = False
        bad_names = ", ".join(sorted(set(non_finite_metrics)))
        error_msg = f"non-finite telemetry values in {bad_names}"
    elif not saw_exporter_up:
        exporter_up = False
        error_msg = "missing freetpu_exporter_up metric"

    return TpuExporterTelemetry(
        reachable=True,
        exporter_up=exporter_up,
        devices_discovered=devices_discovered,
        healthy_chips=healthy_chips,
        total_chips=total_chips,
        max_hbm_utilization_ratio=max(ratios) if ratios else None,
        avg_hbm_utilization_ratio=(sum(ratios) / len(ratios)) if ratios else None,
        hbm_used_bytes=sum(hbm_used_by_chip.values()) if hbm_used_by_chip else None,
        hbm_total_bytes=sum(hbm_total_by_chip.values()) if hbm_total_by_chip else None,
        avg_duty_cycle_percent=(sum(duties) / len(duties)) if duties else None,
        max_duty_cycle_percent=max(duties) if duties else None,
        active_pids=sum(pids_by_chip.values()),
        generations=sorted(generations_seen),
        scraped_at_unix=scraped_at if scraped_at is not None else time.time(),
        error=error_msg,
    )


def evaluate_tpu_readiness(
    telemetry: TpuExporterTelemetry,
    *,
    max_hbm_utilization: float = 0.95,
    max_duty_cycle_percent: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """Return `(is_ready, unready_reason)` for a backend given its TPU telemetry."""
    if not telemetry.reachable:
        return False, "tpu_exporter_unreachable"
    if not telemetry.exporter_up:
        if telemetry.error and (
            telemetry.error == "missing freetpu_exporter_up metric"
            or telemetry.error.startswith("non-finite telemetry")
        ):
            return False, "tpu_exporter_invalid_metrics"
        return False, "tpu_exporter_down"
    if telemetry.total_chips <= 0:
        return False, "tpu_no_devices_discovered"
    if telemetry.healthy_chips == 0:
        return False, "tpu_unhealthy"
    if (
        telemetry.max_hbm_utilization_ratio is not None
        and telemetry.max_hbm_utilization_ratio >= max_hbm_utilization
    ):
        return False, "tpu_hbm_oom_risk"
    if (
        max_duty_cycle_percent is not None
        and telemetry.max_duty_cycle_percent is not None
        and telemetry.max_duty_cycle_percent >= max_duty_cycle_percent
    ):
        return False, "tpu_duty_cycle_overloaded"
    return True, None


def scrape_tpu_exporter(
    url: str,
    *,
    timeout_s: float = 2.0,
) -> TpuExporterTelemetry:
    """Fetch and parse `/metrics` from a `tpu-exporter` HTTP(S) endpoint."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if scheme not in ("http", "https") or not host:
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=f"Invalid tpu_exporter_url: {url!r}",
        )
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/metrics"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    conn: Optional[http.client.HTTPConnection] = None
    try:
        if scheme == "https":
            conn = http.client.HTTPSConnection(
                host, port, timeout=timeout_s, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        conn.request("GET", path, headers={"Accept": "text/plain"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status < 200 or resp.status >= 300:
            return TpuExporterTelemetry(
                reachable=False,
                exporter_up=False,
                devices_discovered=0,
                healthy_chips=0,
                total_chips=0,
                error=f"HTTP {resp.status} from tpu-exporter",
            )
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=str(exc),
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    samples = parse_prometheus_text(body)
    return summarize_tpu_samples(samples)
