#!/usr/bin/env python3
"""Thread-safe Prometheus text exposition 0.0.4 metrics registry for FreeTPU Router."""

from __future__ import annotations

from collections import defaultdict
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

_DEFAULT_TTFT_BUCKETS: Tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

_DEFAULT_LATENCY_BUCKETS: Tuple[float, ...] = (
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


def _esc_label(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _fmt_labels(labels: Dict[str, Any]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_esc_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def _fmt_float(value: float) -> str:
    if value == 0.0:
        return "0"
    rendered = f"{value:.8f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


class _HistogramSeries:
    def __init__(self, buckets: Tuple[float, ...]) -> None:
        self.buckets = buckets
        self.bucket_counts = [0 for _ in buckets]
        self.count = 0
        self.total_sum = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total_sum += value
        for idx, bound in enumerate(self.buckets):
            if value <= bound:
                self.bucket_counts[idx] += 1


class RouterMetrics:
    """Thread-safe metrics collector rendering Prometheus exposition 0.0.4."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._active_inflight: Dict[str, int] = defaultdict(int)
        self._failovers_total: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        self._prompt_tokens_total: Dict[Tuple[str, str], int] = defaultdict(int)
        self._completion_tokens_total: Dict[Tuple[str, str], int] = defaultdict(int)
        self._estimated_cost_usd_total: Dict[Tuple[str, str], float] = defaultdict(float)
        self._ttft_histograms: Dict[Tuple[str, str], _HistogramSeries] = {}
        self._latency_histograms: Dict[Tuple[str, str], _HistogramSeries] = {}
        self._usage_meter_errors_total: int = 0

    def record_usage_meter_error(self) -> int:
        with self._lock:
            self._usage_meter_errors_total += 1
            return self._usage_meter_errors_total

    def usage_meter_errors_total(self) -> int:
        with self._lock:
            return self._usage_meter_errors_total

    def inc_inflight(self, backend_id: str) -> int:
        with self._lock:
            self._active_inflight[backend_id] += 1
            return sum(self._active_inflight.values())

    def dec_inflight(self, backend_id: str) -> int:
        with self._lock:
            cur = self._active_inflight.get(backend_id, 0)
            self._active_inflight[backend_id] = max(0, cur - 1)
            return sum(self._active_inflight.values())

    def total_inflight(self) -> int:
        with self._lock:
            return sum(self._active_inflight.values())

    def record_failover(
        self,
        *,
        model: str,
        from_backend: str,
        to_backend: str,
        reason: str,
    ) -> None:
        with self._lock:
            self._failovers_total[(model, from_backend, to_backend, reason)] += 1

    def record_request(
        self,
        *,
        model: str,
        backend_id: str,
        status: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        ttft_ms: Optional[float],
        estimated_cost_usd: float,
    ) -> None:
        with self._lock:
            self._requests_total[(model, backend_id, str(status))] += 1
            if prompt_tokens > 0:
                self._prompt_tokens_total[(model, backend_id)] += prompt_tokens
            if completion_tokens > 0:
                self._completion_tokens_total[(model, backend_id)] += completion_tokens
            if estimated_cost_usd > 0.0:
                self._estimated_cost_usd_total[(model, backend_id)] += estimated_cost_usd

            key = (model, backend_id)
            lat_series = self._latency_histograms.get(key)
            if lat_series is None:
                lat_series = _HistogramSeries(_DEFAULT_LATENCY_BUCKETS)
                self._latency_histograms[key] = lat_series
            lat_series.observe(max(0.0, latency_ms / 1000.0))

            if ttft_ms is not None:
                ttft_series = self._ttft_histograms.get(key)
                if ttft_series is None:
                    ttft_series = _HistogramSeries(_DEFAULT_TTFT_BUCKETS)
                    self._ttft_histograms[key] = ttft_series
                ttft_series.observe(max(0.0, ttft_ms / 1000.0))

    def render_prometheus(self, backend_snapshots: Iterable[Dict[str, Any]]) -> str:
        """Render router and backend metrics in Prometheus text format 0.0.4."""
        lines: List[str] = []

        def emit_simple(
            name: str,
            help_text: str,
            mtype: str,
            samples: List[Tuple[Dict[str, Any], str]],
        ) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            for labels, value in samples:
                lines.append(f"{name}{_fmt_labels(labels)} {value}")

        def emit_histogram(
            name: str,
            help_text: str,
            series_map: Dict[Tuple[str, str], _HistogramSeries],
        ) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} histogram")
            for (model, backend_id), series in sorted(series_map.items()):
                base = {"model": model, "backend_id": backend_id}
                for bound, cnt in zip(series.buckets, series.bucket_counts):
                    lines.append(
                        f"{name}_bucket{_fmt_labels({**base, 'le': _fmt_float(bound)})} {cnt}"
                    )
                lines.append(
                    f"{name}_bucket{_fmt_labels({**base, 'le': '+Inf'})} {series.count}"
                )
                lines.append(
                    f"{name}_sum{_fmt_labels(base)} {_fmt_float(series.total_sum)}"
                )
                lines.append(f"{name}_count{_fmt_labels(base)} {series.count}")

        with self._lock:
            req_samples = [
                ({"model": m, "backend_id": b, "status": s}, str(cnt))
                for (m, b, s), cnt in sorted(self._requests_total.items())
            ]
            inflight_samples = [
                ({"backend_id": b}, str(cnt))
                for b, cnt in sorted(self._active_inflight.items())
            ]
            failover_samples = [
                (
                    {
                        "model": m,
                        "from_backend": fb,
                        "to_backend": tb,
                        "reason": r,
                    },
                    str(cnt),
                )
                for (m, fb, tb, r), cnt in sorted(self._failovers_total.items())
            ]
            prompt_samples = [
                ({"model": m, "backend_id": b}, str(cnt))
                for (m, b), cnt in sorted(self._prompt_tokens_total.items())
            ]
            completion_samples = [
                ({"model": m, "backend_id": b}, str(cnt))
                for (m, b), cnt in sorted(self._completion_tokens_total.items())
            ]
            cost_samples = [
                ({"model": m, "backend_id": b}, _fmt_float(usd))
                for (m, b), usd in sorted(self._estimated_cost_usd_total.items())
            ]
            ttft_copy = dict(self._ttft_histograms)
            lat_copy = dict(self._latency_histograms)
            usage_meter_errors = self._usage_meter_errors_total

        backend_list = list(backend_snapshots)
        for snap in backend_list:
            bid = str(snap["id"])
            if not any(lbl["backend_id"] == bid for lbl, _ in inflight_samples):
                inflight_samples.append(({"backend_id": bid}, "0"))
        inflight_samples.sort(key=lambda item: item[0]["backend_id"])

        emit_simple(
            "freetpu_router_up",
            "Whether the FreeTPU Router process is up (1 = up).",
            "gauge",
            [({}, "1")],
        )
        emit_simple(
            "freetpu_router_requests_total",
            "Total OpenAI API inference requests handled by FreeTPU Router.",
            "counter",
            req_samples,
        )
        emit_simple(
            "freetpu_router_active_inflight",
            "Current active in-flight inference requests per backend.",
            "gauge",
            inflight_samples,
        )
        emit_simple(
            "freetpu_router_failovers_total",
            "Total automatic failovers across heterogeneous backends (e.g. Spot TPU preemption).",
            "counter",
            failover_samples,
        )
        emit_simple(
            "freetpu_router_prompt_tokens_total",
            "Total prompt tokens processed per model and backend.",
            "counter",
            prompt_samples,
        )
        emit_simple(
            "freetpu_router_completion_tokens_total",
            "Total completion tokens generated per model and backend.",
            "counter",
            completion_samples,
        )
        emit_simple(
            "freetpu_router_estimated_cost_usd_total",
            "Estimated cumulative inference cost in USD per model and backend.",
            "counter",
            cost_samples,
        )
        emit_simple(
            "freetpu_router_usage_meter_errors_total",
            "Total errors encountered while persisting usage metering records.",
            "counter",
            [({}, str(usage_meter_errors))],
        )
        emit_histogram(
            "freetpu_router_ttft_seconds",
            "Time to first token (TTFT) in seconds per model and backend.",
            ttft_copy,
        )
        emit_histogram(
            "freetpu_router_request_duration_seconds",
            "End-to-end request latency in seconds per model and backend.",
            lat_copy,
        )

        health_samples: List[Tuple[Dict[str, Any], str]] = []
        backend_lat_samples: List[Tuple[Dict[str, Any], str]] = []
        hbm_samples: List[Tuple[Dict[str, Any], str]] = []
        duty_samples: List[Tuple[Dict[str, Any], str]] = []
        chip_samples: List[Tuple[Dict[str, Any], str]] = []

        for snap in backend_list:
            bid = str(snap["id"])
            health_samples.append(
                (
                    {
                        "backend_id": bid,
                        "type": str(snap["type"]),
                        "local": "true" if snap["local"] else "false",
                        "spot": "true" if snap["spot"] else "false",
                    },
                    "1" if snap["healthy"] else "0",
                )
            )
            if snap.get("latency_ms") is not None:
                backend_lat_samples.append(
                    ({"backend_id": bid}, _fmt_float(float(snap["latency_ms"]) / 1000.0))
                )
            tpu = snap.get("tpu_telemetry")
            if isinstance(tpu, dict):
                if tpu.get("max_hbm_utilization_ratio") is not None:
                    hbm_samples.append(
                        (
                            {"backend_id": bid},
                            _fmt_float(float(tpu["max_hbm_utilization_ratio"])),
                        )
                    )
                if tpu.get("avg_duty_cycle_percent") is not None:
                    duty_samples.append(
                        (
                            {"backend_id": bid},
                            _fmt_float(float(tpu["avg_duty_cycle_percent"])),
                        )
                    )
                chip_samples.append(
                    ({"backend_id": bid}, str(int(tpu.get("healthy_chips", 0))))
                )

        emit_simple(
            "freetpu_router_backend_healthy",
            "Backend health status as observed by FreeTPU Router (1 = healthy, 0 = unhealthy/overloaded).",
            "gauge",
            health_samples,
        )
        if backend_lat_samples:
            emit_simple(
                "freetpu_router_backend_latency_seconds",
                "Observed EWMA latency per backend in seconds.",
                "gauge",
                backend_lat_samples,
            )
        if hbm_samples:
            emit_simple(
                "freetpu_router_backend_tpu_hbm_utilization_ratio",
                "Latest scraped peak HBM utilization ratio (0.0 - 1.0) per TPU backend.",
                "gauge",
                hbm_samples,
            )
        if duty_samples:
            emit_simple(
                "freetpu_router_backend_tpu_duty_cycle_percent",
                "Latest scraped average MXU duty cycle percent per TPU backend.",
                "gauge",
                duty_samples,
            )
        if chip_samples:
            emit_simple(
                "freetpu_router_backend_tpu_healthy_chips",
                "Number of healthy TPU chips reported by tpu-exporter for each backend.",
                "gauge",
                chip_samples,
            )

        return "\n".join(lines) + "\n"
