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
import errno
import http.client
import io
import math
import re
import select
import socket
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


MAX_EXPORTER_RESPONSE_BYTES = 2 * 1024 * 1024


class ClientDisconnectedError(Exception):
    """Raised when the downstream client closes its connection during an upstream wait."""


_MSG_DONTWAIT = getattr(socket, "MSG_DONTWAIT", 0)


class _DeadlineSocketIO(io.RawIOBase):
    """Raw socket reader enforcing an elapsed deadline and optional client cancellation."""

    def __init__(self, wrapper: "DeadlineSocket") -> None:
        self._wrapper = wrapper
        self._nonblocking_configured = False

    def readable(self) -> bool:
        return True

    def _wait_readable_or_cancel(self) -> None:
        upstream = self._wrapper._sock
        deadline = self._wrapper.deadline
        # If TLS already decrypted buffered bytes, read immediately without blocking in select.
        pending_fn = getattr(upstream, "pending", None)
        if callable(pending_fn) and pending_fn() > 0:
            return
        while True:
            rem: Optional[float] = None
            if deadline is not None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"{self._wrapper.label} exceeded elapsed deadline")
            if self._wrapper._watch_client and self._wrapper._client_sock is not None:
                client = self._wrapper._client_sock
                try:
                    readable, _, errored = select.select(
                        [upstream, client], [], [client], rem
                    )
                except (OSError, ValueError):
                    try:
                        if client.fileno() == -1:
                            raise ClientDisconnectedError("Downstream client disconnected")
                    except (OSError, AttributeError):
                        raise ClientDisconnectedError("Downstream client disconnected")
                    self._wrapper._watch_client = False
                    continue
                if client in errored:
                    raise ClientDisconnectedError("Downstream client disconnected")
                if client in readable:
                    peek: Optional[bytes] = None
                    try:
                        peek = client.recv(
                            1, socket.MSG_PEEK | _MSG_DONTWAIT
                        )
                    except BlockingIOError:
                        peek = None
                    except OSError as exc:
                        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            peek = None
                        else:
                            raise ClientDisconnectedError("Downstream client disconnected") from exc
                    if peek == b"":
                        raise ClientDisconnectedError("Downstream client disconnected")
                    if peek is not None:
                        # Extra pipelined bytes on client socket; stop EOF-watching to avoid busy spin.
                        self._wrapper._watch_client = False
                if upstream in readable:
                    return
                if not readable and not errored and deadline is not None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{self._wrapper.label} exceeded elapsed deadline")
            else:
                readable, _, _ = select.select([upstream], [], [], rem)
                if upstream in readable:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"{self._wrapper.label} exceeded elapsed deadline")

    def readinto(self, b: Any) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        wrapper = self._wrapper
        upstream = wrapper._sock
        deadline = wrapper.deadline
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{wrapper.label} exceeded elapsed deadline")

        if not wrapper._is_ssl:
            if not self._nonblocking_configured:
                try:
                    upstream.settimeout(0.0)
                    self._nonblocking_configured = True
                except OSError:
                    pass
            if self._nonblocking_configured:
                while True:
                    try:
                        n = upstream.recv_into(b)
                        if (
                            n > 0
                            and wrapper._watch_client
                            and wrapper._check_client_on_fast_read
                            and wrapper._client_sock is not None
                        ):
                            client = wrapper._client_sock
                            try:
                                r_cli, _, e_cli = select.select([client], [], [client], 0.0)
                                if e_cli:
                                    raise ClientDisconnectedError("Downstream client disconnected")
                                if r_cli:
                                    peek = client.recv(1, socket.MSG_PEEK | _MSG_DONTWAIT)
                                    if peek == b"":
                                        raise ClientDisconnectedError("Downstream client disconnected")
                                    wrapper._watch_client = False
                            except ClientDisconnectedError:
                                raise
                            except (OSError, ValueError):
                                pass
                        return n
                    except BlockingIOError:
                        pass
                    except OSError as exc:
                        if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                            raise
                    self._wait_readable_or_cancel()

        self._wait_readable_or_cancel()
        if deadline is not None:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"{wrapper.label} exceeded elapsed deadline")
            upstream.settimeout(rem)
        return upstream.recv_into(b)

    def close(self) -> None:
        if not self.closed:
            super().close()
            self._wrapper._decref_io()


class DeadlineSocket:
    """Socket wrapper enforcing an elapsed deadline and optional downstream disconnect cancellation."""

    def __init__(
        self,
        sock: Any,
        deadline: Optional[float],
        *,
        label: str = "HTTP response",
        client_sock: Any = None,
    ) -> None:
        self._sock = sock
        self.deadline = deadline
        self.label = label
        self._client_sock = client_sock
        self._watch_client = client_sock is not None
        self._check_client_on_fast_read = client_sock is not None
        self._is_ssl = isinstance(sock, ssl.SSLSocket)
        self._io_refs = 0
        self._closed = False

    def makefile(
        self,
        mode: str = "r",
        buffering: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "b" in mode and "r" in mode and "w" not in mode and "+" not in mode:
            self._io_refs += 1
            raw = _DeadlineSocketIO(self)
            buf_size = (
                buffering
                if (isinstance(buffering, int) and buffering > 0)
                else io.DEFAULT_BUFFER_SIZE
            )
            return io.BufferedReader(raw, buffer_size=buf_size)
        return self._sock.makefile(mode, buffering, *args, **kwargs)

    def settimeout(self, timeout: Optional[float]) -> None:
        if self.deadline is not None:
            rem = self.deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"{self.label} exceeded elapsed deadline")
            effective = min(timeout, rem) if timeout is not None else rem
            self._sock.settimeout(effective)
        else:
            self._sock.settimeout(timeout)

    def _decref_io(self) -> None:
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._io_refs == 0 and self._closed:
            self._sock.close()

    def close(self) -> None:
        self._closed = True
        if self._io_refs <= 0:
            self._sock.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


def _read_bounded_http_body(
    resp: http.client.HTTPResponse,
    upstream_sock: Any,
    *,
    deadline: float,
    max_bytes: int = MAX_EXPORTER_RESPONSE_BYTES,
) -> bytes:
    """Read an HTTP response body incrementally under an elapsed wall-clock deadline."""
    declared_len: Optional[int] = None
    if not resp.chunked:
        raw_len = resp.getheader("Content-Length")
        if raw_len is not None:
            try:
                parsed_len = int(raw_len.strip())
            except ValueError:
                parsed_len = -1
            if parsed_len >= 0:
                declared_len = parsed_len
                if declared_len > max_bytes:
                    raise ValueError(
                        f"Exporter response exceeded {max_bytes} byte limit"
                    )
    chunks: List[bytes] = []
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Exporter scrape exceeded elapsed deadline")
        if upstream_sock is not None:
            try:
                upstream_sock.settimeout(remaining)
            except OSError:
                pass
        budget_left = (max_bytes - total) + 1
        if budget_left <= 0:
            raise ValueError(f"Exporter response exceeded {max_bytes} byte limit")
        chunk = resp.read1(min(8192, budget_left))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Exporter response exceeded {max_bytes} byte limit")
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not resp.chunked and resp.status not in (204, 304) and not (100 <= resp.status < 200):
        if resp.length is not None and resp.length > 0:
            raise http.client.IncompleteRead(raw, resp.length)
        if declared_len is not None and len(raw) < declared_len:
            raise http.client.IncompleteRead(raw, declared_len - len(raw))
    return raw


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
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=f"Invalid tpu_exporter_url port: {exc}",
        )
    if parsed_port is not None and not (1 <= parsed_port <= 65535):
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=f"Invalid tpu_exporter_url port: {parsed_port}",
        )
    port = parsed_port or (443 if scheme == "https" else 80)
    path = parsed.path or "/metrics"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    deadline = time.monotonic() + max(0.01, timeout_s)
    conn: Optional[http.client.HTTPConnection] = None
    resp: Optional[http.client.HTTPResponse] = None
    raw_sock: Any = None
    try:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("Exporter scrape exceeded elapsed deadline")
        if scheme == "https":
            conn = http.client.HTTPSConnection(
                host, port, timeout=rem, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=rem)
        conn.request("GET", path, headers={"Accept": "text/plain"})
        raw_sock = conn.sock
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("Exporter scrape exceeded elapsed deadline")
        if raw_sock is not None:
            conn.sock = DeadlineSocket(raw_sock, deadline, label="Exporter scrape")
        resp = conn.getresponse()
        upstream_sock = raw_sock
        if resp.status < 200 or resp.status >= 300:
            return TpuExporterTelemetry(
                reachable=False,
                exporter_up=False,
                devices_discovered=0,
                healthy_chips=0,
                total_chips=0,
                error=f"HTTP {resp.status} from tpu-exporter",
            )
        body = _read_bounded_http_body(
            resp, upstream_sock, deadline=deadline
        ).decode("utf-8", errors="replace")
    except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
        if isinstance(exc, TimeoutError):
            safe_err = "exporter_scrape_timeout"
        elif isinstance(exc, ssl.SSLError):
            safe_err = "exporter_tls_error"
        elif isinstance(exc, http.client.IncompleteRead):
            safe_err = "exporter_incomplete_response"
        elif isinstance(exc, http.client.HTTPException):
            safe_err = f"exporter_http_error:{type(exc).__name__}"
        elif isinstance(exc, ConnectionRefusedError):
            safe_err = "exporter_connection_refused"
        elif isinstance(exc, OSError):
            safe_err = f"exporter_network_error:{type(exc).__name__}"
        else:
            safe_err = f"exporter_error:{type(exc).__name__}"
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=safe_err,
        )
    finally:
        if resp is not None:
            try:
                resp.close()
            except OSError:
                pass
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if raw_sock is not None:
            try:
                raw_sock.close()
            except OSError:
                pass

    samples = parse_prometheus_text(body)
    return summarize_tpu_samples(samples)
