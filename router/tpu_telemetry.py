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
import ipaddress
import math
import re
import select
import selectors
import socket
import ssl
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)(?:\s+[+-]?[0-9]+)?$"
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
        if "_" in val_str:
            continue
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
_POLL_READ_EVENTS = getattr(select, "POLLIN", 0) | getattr(select, "POLLPRI", 0)
_POLL_HUP_EVENT = getattr(select, "POLLHUP", 0)
_POLL_ERR_EVENTS = getattr(select, "POLLERR", 0) | getattr(select, "POLLNVAL", 0)


def wait_sockets_readable(
    sockets: Sequence[Any],
    timeout_s: Optional[float],
) -> Tuple[set[Any], set[Any]]:
    """Wait for socket readability or error using an API supporting FDs >= 1024.

    Returns `(readable, errored)` subsets of `sockets`.
    """
    if hasattr(select, "poll"):
        poller = select.poll()
        fd_to_socks: Dict[int, List[Any]] = {}
        for sock in sockets:
            fd = sock.fileno()
            if fd < 0:
                raise ValueError("Invalid socket file descriptor")
            if fd not in fd_to_socks:
                fd_to_socks[fd] = [sock]
                poller.register(fd, _POLL_READ_EVENTS)
            else:
                fd_to_socks[fd].append(sock)
        if timeout_s is None:
            timeout_ms: Optional[int] = None
        elif timeout_s <= 0:
            timeout_ms = 0
        else:
            timeout_ms = min(2147483647, max(1, math.ceil(timeout_s * 1000.0)))
        ready = poller.poll(timeout_ms)
        readable: set[Any] = set()
        errored: set[Any] = set()
        for fd, revents in ready:
            matched = fd_to_socks.get(fd)
            if not matched:
                continue
            if revents & (_POLL_READ_EVENTS | _POLL_HUP_EVENT):
                readable.update(matched)
            if revents & _POLL_ERR_EVENTS:
                errored.update(matched)
        return readable, errored

    with selectors.DefaultSelector() as selector:
        for sock in sockets:
            selector.register(sock, selectors.EVENT_READ, data=sock)
        events = selector.select(timeout=timeout_s)
        return {key.data for key, _ in events}, set()


def _wait_socket_writable(sock: Any, timeout_s: Optional[float]) -> bool:
    """Wait until `sock` is writable or errored, supporting FDs >= 1024."""
    if hasattr(select, "poll"):
        poller = select.poll()
        fd = sock.fileno()
        if fd < 0:
            raise ValueError("Invalid socket file descriptor")
        poller.register(fd, select.POLLOUT)
        if timeout_s is None:
            timeout_ms: Optional[int] = None
        elif timeout_s <= 0:
            timeout_ms = 0
        else:
            timeout_ms = min(2147483647, max(1, math.ceil(timeout_s * 1000.0)))
        return bool(poller.poll(timeout_ms))

    with selectors.DefaultSelector() as selector:
        selector.register(sock, selectors.EVENT_WRITE, data=sock)
        return bool(selector.select(timeout=timeout_s))


def _tls_handshake_with_deadline(
    tls_sock: ssl.SSLSocket,
    *,
    deadline: float,
    label: str,
) -> None:
    """Drive a nonblocking TLS handshake strictly within `deadline`."""
    orig_timeout = tls_sock.gettimeout()
    tls_sock.setblocking(False)
    try:
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"{label} exceeded elapsed deadline")
            try:
                tls_sock.do_handshake()
                return
            except ssl.SSLWantReadError:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"{label} exceeded elapsed deadline")
                wait_sockets_readable([tls_sock], rem)
            except ssl.SSLWantWriteError:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"{label} exceeded elapsed deadline")
                _wait_socket_writable(tls_sock, rem)
    finally:
        try:
            rem = deadline - time.monotonic()
            if orig_timeout is None:
                tls_sock.settimeout(None)
            elif rem > 0:
                tls_sock.settimeout(rem)
            else:
                tls_sock.settimeout(0.0)
        except OSError:
            pass


def _wait_io_or_client(
    upstream: Any,
    client: Any,
    *,
    wait_write: bool,
    timeout_s: Optional[float],
) -> Tuple[bool, bool, bool]:
    """Wait for `upstream` readability/writability while optionally watching `client` readability.

    Returns `(upstream_ready, client_readable, client_errored)` using an API that
    supports file descriptors >= 1024.
    """
    if hasattr(select, "poll"):
        poller = select.poll()
        up_fd = upstream.fileno()
        if up_fd < 0:
            raise ValueError("Invalid upstream socket file descriptor")
        up_mask = select.POLLOUT if wait_write else _POLL_READ_EVENTS
        poller.register(up_fd, up_mask)
        cli_fd: Optional[int] = None
        if client is not None:
            cli_fd = client.fileno()
            if cli_fd < 0:
                raise ClientDisconnectedError("Downstream client disconnected")
            if cli_fd == up_fd:
                poller.modify(up_fd, up_mask | _POLL_READ_EVENTS)
            else:
                poller.register(cli_fd, _POLL_READ_EVENTS)
        if timeout_s is None:
            timeout_ms: Optional[int] = None
        elif timeout_s <= 0:
            timeout_ms = 0
        else:
            timeout_ms = min(2147483647, max(1, math.ceil(timeout_s * 1000.0)))
        ready = poller.poll(timeout_ms)
        up_ready = False
        cli_read = False
        cli_err = False
        for fd, revents in ready:
            if fd == up_fd and (revents & (up_mask | _POLL_HUP_EVENT | _POLL_ERR_EVENTS)):
                up_ready = True
            if cli_fd is not None and fd == cli_fd:
                if revents & _POLL_ERR_EVENTS:
                    cli_err = True
                if revents & (_POLL_READ_EVENTS | _POLL_HUP_EVENT):
                    cli_read = True
        return up_ready, cli_read, cli_err

    with selectors.DefaultSelector() as selector:
        up_ev = selectors.EVENT_WRITE if wait_write else selectors.EVENT_READ
        selector.register(upstream, up_ev, data="up")
        if client is not None:
            selector.register(client, selectors.EVENT_READ, data="cli")
        events = selector.select(timeout=timeout_s)
        up_ready = False
        cli_read = False
        for key, _ in events:
            if key.data == "up":
                up_ready = True
            elif key.data == "cli":
                cli_read = True
        return up_ready, cli_read, False


_DNS_RESOLVER_SEMAPHORE = threading.BoundedSemaphore(32)


def _resolve_host_with_deadline(
    host: str,
    port: int,
    *,
    deadline: float,
    label: str = "Exporter scrape",
) -> List[Any]:
    """Resolve `host` via `socket.getaddrinfo` bounded by `deadline`."""
    rem = deadline - time.monotonic()
    if rem <= 0:
        raise TimeoutError(f"{label} exceeded elapsed deadline")
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False
    if is_ip_literal:
        infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        return infos

    if not _DNS_RESOLVER_SEMAPHORE.acquire(timeout=rem):
        raise TimeoutError(f"{label} exceeded elapsed deadline")

    done = threading.Event()
    result_box: List[Any] = []
    err_box: List[BaseException] = []
    getaddrinfo_fn = socket.getaddrinfo

    def _worker() -> None:
        try:
            result_box.append(getaddrinfo_fn(host, port, 0, socket.SOCK_STREAM))
        except BaseException as exc:  # noqa: BLE001
            err_box.append(exc)
        finally:
            done.set()
            _DNS_RESOLVER_SEMAPHORE.release()

    try:
        thread = threading.Thread(
            target=_worker,
            name="freetpu-dns-resolve",
            daemon=True,
        )
        thread.start()
    except BaseException:
        _DNS_RESOLVER_SEMAPHORE.release()
        raise

    rem = deadline - time.monotonic()
    if rem <= 0 or not done.wait(timeout=rem):
        raise TimeoutError(f"{label} exceeded elapsed deadline")
    if err_box:
        raise err_box[0]
    return result_box[0] if result_box else []


def _create_deadline_connection(
    host: str,
    port: int,
    *,
    deadline: float,
    label: str = "Exporter scrape",
    ssl_context: Optional[ssl.SSLContext] = None,
    server_hostname: Optional[str] = None,
) -> socket.socket:
    """Resolve `host`, connect across candidate addresses, and run TLS under one deadline."""
    infos = _resolve_host_with_deadline(host, port, deadline=deadline, label=label)
    if not infos:
        raise OSError("getaddrinfo returns an empty list")

    last_err: Optional[OSError] = None
    sock: Optional[socket.socket] = None
    for af, socktype, proto, _canonname, sa in infos:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        cand_sock: Optional[socket.socket] = None
        try:
            cand_sock = socket.socket(af, socktype, proto)
            cand_sock.settimeout(rem)
            cand_sock.connect(sa)
            sock = cand_sock
            last_err = None
            break
        except OSError as exc:
            last_err = exc
            if cand_sock is not None:
                try:
                    cand_sock.close()
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label} exceeded elapsed deadline") from exc

    if sock is None:
        if last_err is not None:
            raise last_err
        raise OSError("getaddrinfo returns an empty list")

    if ssl_context is not None:
        rem = deadline - time.monotonic()
        if rem <= 0:
            try:
                sock.close()
            except OSError:
                pass
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        tls_sock: Optional[ssl.SSLSocket] = None
        try:
            sock.settimeout(rem)
            tls_sock = ssl_context.wrap_socket(
                sock,
                server_hostname=server_hostname or host,
                do_handshake_on_connect=False,
            )
            _tls_handshake_with_deadline(tls_sock, deadline=deadline, label=label)
            sock = tls_sock
        except BaseException:
            if tls_sock is not None:
                try:
                    tls_sock.close()
                except OSError:
                    pass
            else:
                try:
                    sock.close()
                except OSError:
                    pass
            raise

    rem = deadline - time.monotonic()
    if rem <= 0:
        try:
            sock.close()
        except OSError:
            pass
        raise TimeoutError(f"{label} exceeded elapsed deadline")
    sock.settimeout(rem)
    return sock


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    """`HTTPConnection` enforcing a single wall-clock deadline across connect setup."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        deadline: float,
        label: str = "Exporter scrape",
    ) -> None:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        super().__init__(host, port, timeout=rem)
        self._deadline = deadline
        self._label = label

    def connect(self) -> None:
        self.sock = _create_deadline_connection(
            self.host,
            self.port,
            deadline=self._deadline,
            label=self._label,
        )


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    """`HTTPSConnection` enforcing a single wall-clock deadline across TCP connect and TLS."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        deadline: float,
        context: ssl.SSLContext,
        label: str = "Exporter scrape",
    ) -> None:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        super().__init__(host, port, timeout=rem, context=context)
        self._deadline = deadline
        self._label = label

    def connect(self) -> None:
        self.sock = _create_deadline_connection(
            self.host,
            self.port,
            deadline=self._deadline,
            label=self._label,
            ssl_context=self._context,
            server_hostname=self.host,
        )


class _DeadlineSocketIO(io.RawIOBase):
    """Raw socket reader enforcing an elapsed deadline and optional client cancellation."""

    def __init__(self, wrapper: "DeadlineSocket") -> None:
        self._wrapper = wrapper
        self._nonblocking_configured = False

    def readable(self) -> bool:
        return True

    def _wait_upstream_or_cancel(
        self,
        *,
        wait_write: bool = False,
        check_pending: bool = True,
    ) -> None:
        upstream = self._wrapper._sock
        deadline = self._wrapper.deadline
        if check_pending and not wait_write:
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
                    up_ready, cli_read, cli_err = _wait_io_or_client(
                        upstream,
                        client,
                        wait_write=wait_write,
                        timeout_s=rem,
                    )
                except ClientDisconnectedError:
                    raise
                except (OSError, ValueError):
                    try:
                        if client.fileno() < 0:
                            raise ClientDisconnectedError("Downstream client disconnected")
                    except (OSError, AttributeError):
                        raise ClientDisconnectedError("Downstream client disconnected")
                    try:
                        if upstream.fileno() < 0:
                            raise
                    except (OSError, AttributeError):
                        raise
                    self._wrapper._watch_client = False
                    continue
                if cli_err:
                    raise ClientDisconnectedError("Downstream client disconnected")
                if cli_read:
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
                if up_ready:
                    return
                if not up_ready and not cli_read and not cli_err and deadline is not None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{self._wrapper.label} exceeded elapsed deadline")
            else:
                up_ready, _, _ = _wait_io_or_client(
                    upstream,
                    None,
                    wait_write=wait_write,
                    timeout_s=rem,
                )
                if up_ready:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"{self._wrapper.label} exceeded elapsed deadline")

    def _wait_readable_or_cancel(self) -> None:
        self._wait_upstream_or_cancel(wait_write=False, check_pending=True)

    def readinto(self, b: Any) -> Optional[int]:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        wrapper = self._wrapper
        if wrapper._buffered_peek_only:
            return None
        upstream = wrapper._sock
        deadline = wrapper.deadline
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{wrapper.label} exceeded elapsed deadline")

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
                            r_cli, e_cli = wait_sockets_readable([client], 0.0)
                            if client in e_cli:
                                raise ClientDisconnectedError("Downstream client disconnected")
                            if client in r_cli:
                                peek = client.recv(1, socket.MSG_PEEK | _MSG_DONTWAIT)
                                if peek == b"":
                                    raise ClientDisconnectedError("Downstream client disconnected")
                                wrapper._watch_client = False
                        except ClientDisconnectedError:
                            raise
                        except (OSError, ValueError):
                            pass
                    return n
                except ssl.SSLWantReadError:
                    self._wait_upstream_or_cancel(wait_write=False, check_pending=False)
                except ssl.SSLWantWriteError:
                    self._wait_upstream_or_cancel(wait_write=True, check_pending=False)
                except BlockingIOError:
                    self._wait_upstream_or_cancel(wait_write=False, check_pending=False)
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    self._wait_upstream_or_cancel(wait_write=False, check_pending=False)

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
        self._buffered_peek_only = False
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
            if self._io_refs > 0:
                return
            effective = min(timeout, rem) if timeout is not None else rem
            self._sock.settimeout(effective)
        else:
            if self._io_refs > 0:
                return
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
        if isinstance(upstream_sock, DeadlineSocket):
            upstream_sock.deadline = deadline
        elif upstream_sock is not None:
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
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname
    except ValueError as exc:
        return TpuExporterTelemetry(
            reachable=False,
            exporter_up=False,
            devices_discovered=0,
            healthy_chips=0,
            total_chips=0,
            error=f"Invalid tpu_exporter_url: {exc}",
        )
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
        if scheme == "https":
            conn = _DeadlineHTTPSConnection(
                host,
                port,
                deadline=deadline,
                context=ssl.create_default_context(),
                label="Exporter scrape",
            )
        else:
            conn = _DeadlineHTTPConnection(
                host,
                port,
                deadline=deadline,
                label="Exporter scrape",
            )
        conn.connect()
        raw_sock = conn.sock
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("Exporter scrape exceeded elapsed deadline")
        if raw_sock is not None:
            raw_sock.settimeout(rem)
        conn.request("GET", path, headers={"Accept": "text/plain"})
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("Exporter scrape exceeded elapsed deadline")
        deadline_sock: Optional[DeadlineSocket] = None
        if raw_sock is not None:
            deadline_sock = DeadlineSocket(raw_sock, deadline, label="Exporter scrape")
            conn.sock = deadline_sock
        resp = conn.getresponse()
        upstream_sock = deadline_sock if deadline_sock is not None else raw_sock
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
