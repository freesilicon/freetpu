#!/usr/bin/env python3
"""Offline end-to-end integration and unit test suite for FreeTPU Router v0."""

from __future__ import annotations

import asyncio
import gc
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import select
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest import mock
from contextlib import nullcontext, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.config import (  # noqa: E402
    BackendConfig,
    RouterConfig,
    load_config,
    sanitize_url,
    validate_http_url,
)
from router.freetpu_router import (  # noqa: E402
    FreeTPURouterServer,
    MAX_RR_CURSORS,
    _MemoryBudgetExceededError,
    _apply_cli_overrides,
    _build_cli_parser,
    _estimate_json_parse_memory,
    _join_backend_url,
    _open_http_connection,
    _read_bounded_http_response,
    _safe_nonneg_int,
    _sse_event_should_commit,
    _validate_json_structure,
    build_hooks_from_config,
    create_router_server,
    main as router_main,
)
from router.hooks import (  # noqa: E402
    BackendIdleEvent,
    BearerTokenAuthHook,
    JsonlUsageMeterHook,
    QueuePressureEvent,
    ScalerHook,
    UsageEvent,
)
from router.tpu_telemetry import (  # noqa: E402
    ClientDisconnectedError,
    DeadlineSocket,
    _read_bounded_http_body,
    evaluate_tpu_readiness,
    parse_prometheus_text,
    scrape_tpu_exporter,
    summarize_tpu_samples,
    wait_sockets_readable,
)


def _abort_tcp_socket(sock: socket.socket) -> None:
    """Abort a TCP connection immediately with RST (distinguishing abort from SHUT_WR FIN)."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _tcp_socketpair() -> Tuple[socket.socket, socket.socket]:
    """Return a connected loopback TCP `(server_side, client_side)` socket pair."""
    srv = socket.create_server(("127.0.0.1", 0))
    try:
        port = srv.getsockname()[1]
        cli = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        acc, _ = srv.accept()
        return acc, cli
    finally:
        srv.close()


class _MockOpenAIBackend(ThreadingHTTPServer):
    """Real local HTTP server simulating an OpenAI-compatible TPU/llama-server worker."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        backend_name: str,
        models: List[str],
        *,
        status_code: int = 200,
        latency_s: float = 0.0,
        abort_stream_before_first_token: bool = False,
        raw_sse_bytes: Optional[bytes] = None,
        custom_sse_chunks: Optional[List[Dict[str, Any]]] = None,
        chunked_keepalive_sse: bool = False,
        keepalive_hold_event: Optional[threading.Event] = None,
        keepalive_storm_interval_s: Optional[float] = None,
        stalled_error_body: bool = False,
        stalled_error_release_event: Optional[threading.Event] = None,
        newline_free_trickle_interval_s: Optional[float] = None,
        post_token_keepalive_interval_s: Optional[float] = None,
        hold_after_done_event: Optional[threading.Event] = None,
        slow_drip_probe_interval_s: Optional[float] = None,
        custom_non_stream_payload: Optional[Dict[str, Any]] = None,
        oversized_non_stream_bytes: Optional[int] = None,
        oversized_non_stream_chunked: bool = False,
        slow_drip_non_stream_interval_s: Optional[float] = None,
        truncate_fixed_length_response: bool = False,
        close_delimited_truncated_json: bool = False,
        raw_models_response: Optional[Tuple[int, str, bytes]] = None,
        raw_health_response: Optional[Tuple[int, str, bytes]] = None,
        prompt_tokens: int = 12,
        completion_tokens: int = 8,
    ) -> None:
        self.backend_name = backend_name
        self.models = models
        self.status_code = status_code
        self.latency_s = latency_s
        self.abort_stream_before_first_token = abort_stream_before_first_token
        self.raw_sse_bytes = raw_sse_bytes
        self.custom_sse_chunks = custom_sse_chunks
        self.chunked_keepalive_sse = chunked_keepalive_sse
        self.keepalive_hold_event = keepalive_hold_event
        self.keepalive_storm_interval_s = keepalive_storm_interval_s
        self.stalled_error_body = stalled_error_body
        self.stalled_error_release_event = stalled_error_release_event
        self.newline_free_trickle_interval_s = newline_free_trickle_interval_s
        self.post_token_keepalive_interval_s = post_token_keepalive_interval_s
        self.hold_after_done_event = hold_after_done_event
        self.slow_drip_probe_interval_s = slow_drip_probe_interval_s
        self.custom_non_stream_payload = custom_non_stream_payload
        self.oversized_non_stream_bytes = oversized_non_stream_bytes
        self.oversized_non_stream_chunked = oversized_non_stream_chunked
        self.slow_drip_non_stream_interval_s = slow_drip_non_stream_interval_s
        self.truncate_fixed_length_response = truncate_fixed_length_response
        self.close_delimited_truncated_json = close_delimited_truncated_json
        self.raw_models_response = raw_models_response
        self.raw_health_response = raw_health_response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests_seen: List[Dict[str, Any]] = []
        self.get_paths_seen: List[str] = []
        self.current_inflight = 0
        self.peak_inflight = 0
        self._lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _MockOpenAIHandler)

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    server: _MockOpenAIBackend
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    def _send_json(self, status: int, payload: Any) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        with self.server._lock:
            self.server.get_paths_seen.append(self.path)
        if self.path == "/health":
            if self.server.raw_health_response is not None:
                st, ctype, raw_body = self.server.raw_health_response
                self.send_response(st)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw_body)
                self.wfile.flush()
                return
            if self.server.status_code >= 500:
                self._send_json(
                    self.server.status_code,
                    {"error": {"message": "worker unhealthy"}},
                )
                return
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            if self.server.raw_models_response is not None:
                st, ctype, raw_body = self.server.raw_models_response
                self.send_response(st)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw_body)
                self.wfile.flush()
                return
            if self.server.slow_drip_probe_interval_s is not None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                deadline = time.monotonic() + 3.0
                try:
                    self.wfile.write(b"{")
                    self.wfile.flush()
                    while time.monotonic() < deadline:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(self.server.slow_drip_probe_interval_s)
                except OSError:
                    pass
                return
            if self.server.status_code >= 500:
                self._send_json(
                    self.server.status_code,
                    {"error": {"message": "worker preempted"}},
                )
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": self.server.backend_name}
                        for m in self.server.models
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        with self.server._lock:
            self.server.requests_seen.append(
                {"path": self.path, "payload": payload}
            )
            self.server.current_inflight += 1
            if self.server.current_inflight > self.server.peak_inflight:
                self.server.peak_inflight = self.server.current_inflight

        try:
            self._handle_post(payload)
        finally:
            with self.server._lock:
                self.server.current_inflight = max(0, self.server.current_inflight - 1)

    def _write_chunk(self, data: bytes) -> None:
        header = f"{len(data):x}\r\n".encode("ascii")
        self.wfile.write(header + data + b"\r\n")
        self.wfile.flush()

    def _handle_post(self, payload: Dict[str, Any]) -> None:
        if self.server.latency_s > 0:
            time.sleep(self.server.latency_s)

        if self.server.status_code != 200:
            if self.server.stalled_error_body:
                self.send_response(self.server.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "4096")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(b'{"error":"stalled')
                    self.wfile.flush()
                except OSError:
                    return
                if self.server.stalled_error_release_event is not None:
                    self.server.stalled_error_release_event.wait(timeout=3.0)
                else:
                    time.sleep(3.0)
                return
            self._send_json(
                self.server.status_code,
                {"error": {"message": f"simulated {self.server.status_code} on {self.server.backend_name}"}},
            )
            return

        model = payload.get("model", self.server.models[0] if self.server.models else "mock")
        stream = bool(payload.get("stream", False))

        if stream:
            if self.server.chunked_keepalive_sse:
                self.close_connection = False
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                chunk1 = json.dumps(
                    {
                        "id": "chatcmpl-chunked",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "split-"}, "finish_reason": None}],
                    }
                )
                chunk2 = json.dumps(
                    {
                        "id": "chatcmpl-chunked",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": self.server.prompt_tokens,
                            "completion_tokens": self.server.completion_tokens,
                        },
                    }
                )
                ev1 = f"data: {chunk1}\n\n".encode("utf-8")
                # Intentionally split the first SSE event across two HTTP chunks.
                mid = len(ev1) // 2
                self._write_chunk(ev1[:mid])
                self._write_chunk(ev1[mid:])
                self._write_chunk(f"data: {chunk2}\n\ndata: [DONE]\n\n".encode("utf-8"))
                # Terminating zero-length chunk while keeping the TCP socket open.
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if self.server.keepalive_hold_event is not None:
                    self.server.keepalive_hold_event.wait(timeout=3.0)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.server.post_token_keepalive_interval_s is not None:
                first_tok = json.dumps(
                    {
                        "id": "chatcmpl-stall",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                        ],
                    }
                )
                deadline = time.monotonic() + 3.0
                try:
                    self.wfile.write(f"data: {first_tok}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    while time.monotonic() < deadline:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        time.sleep(self.server.post_token_keepalive_interval_s)
                except OSError:
                    pass
                return
            if self.server.newline_free_trickle_interval_s is not None:
                deadline = time.monotonic() + 3.0
                try:
                    self.wfile.write(b":")
                    self.wfile.flush()
                    while time.monotonic() < deadline:
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(self.server.newline_free_trickle_interval_s)
                except OSError:
                    pass
                return
            if self.server.keepalive_storm_interval_s is not None:
                deadline = time.monotonic() + 3.0
                try:
                    while time.monotonic() < deadline:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        time.sleep(self.server.keepalive_storm_interval_s)
                except OSError:
                    pass
                return
            if self.server.raw_sse_bytes is not None:
                if self.server.raw_sse_bytes:
                    self.wfile.write(self.server.raw_sse_bytes)
                self.wfile.flush()
                return
            if self.server.abort_stream_before_first_token:
                self.wfile.flush()
                return
            chunks = (
                self.server.custom_sse_chunks
                if self.server.custom_sse_chunks is not None
                else [
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": self.server.backend_name}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": self.server.prompt_tokens,
                            "completion_tokens": self.server.completion_tokens,
                            "total_tokens": self.server.prompt_tokens + self.server.completion_tokens,
                        },
                    },
                ]
            )
            for chunk in chunks:
                line = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                self.wfile.write(line)
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if self.server.hold_after_done_event is not None:
                self.server.hold_after_done_event.wait(timeout=3.0)
            return

        if self.path == "/v1/completions":
            self._send_json(
                200,
                {
                    "id": "cmpl-mock",
                    "object": "text_completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": f"completion-from:{self.server.backend_name}",
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": self.server.prompt_tokens,
                        "completion_tokens": self.server.completion_tokens,
                        "total_tokens": self.server.prompt_tokens + self.server.completion_tokens,
                    },
                },
            )
            return

        if self.server.close_delimited_truncated_json:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(b'{"id":"cmpl","choices":[')
                self.wfile.flush()
            except OSError:
                pass
            return

        if self.server.truncate_fixed_length_response:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "200")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(b'{"id":"chatcmpl-trun')
                self.wfile.flush()
            except OSError:
                pass
            return

        if self.server.slow_drip_non_stream_interval_s is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            deadline = time.monotonic() + 3.0
            try:
                self.wfile.write(b"{")
                self.wfile.flush()
                while time.monotonic() < deadline:
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(self.server.slow_drip_non_stream_interval_s)
            except OSError:
                pass
            return

        if self.server.oversized_non_stream_bytes is not None:
            pad = "x" * self.server.oversized_non_stream_bytes
            body = json.dumps({"choices": [{"message": {"content": pad}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if self.server.oversized_non_stream_chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for off in range(0, len(body), 512):
                        self._write_chunk(body[off : off + 512])
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass
            else:
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                except OSError:
                    pass
            return

        if self.server.custom_non_stream_payload is not None:
            self._send_json(200, self.server.custom_non_stream_payload)
            return

        self._send_json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"reply-from:{self.server.backend_name}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": self.server.prompt_tokens,
                    "completion_tokens": self.server.completion_tokens,
                    "total_tokens": self.server.prompt_tokens + self.server.completion_tokens,
                },
            },
        )


class _MockTpuExporter(ThreadingHTTPServer):
    """Mock Prometheus `tpu-exporter` emitting exact PR #9 metric names."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        *,
        exporter_up: int = 1,
        health_status: int = 1,
        hbm_used_bytes: int = 4 * 1024**3,
        hbm_total_bytes: int = 16 * 1024**3,
        duty_cycle_percent: float = 25.0,
        generation: str = "v5e",
        raw_body: Optional[str] = None,
    ) -> None:
        self.exporter_up = exporter_up
        self.health_status = health_status
        self.hbm_used_bytes = hbm_used_bytes
        self.hbm_total_bytes = hbm_total_bytes
        self.duty_cycle_percent = duty_cycle_percent
        self.generation = generation
        self.raw_body = raw_body
        super().__init__(("127.0.0.1", 0), _MockTpuExporterHandler)

    @property
    def metrics_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/metrics"

    def render(self) -> str:
        if self.raw_body is not None:
            return self.raw_body
        ratio = self.hbm_used_bytes / float(self.hbm_total_bytes)
        labels = (
            f'chip="0",pci_bdf="0000:00:04.0",device_id="0x0063",'
            f'generation="{self.generation}",vfio_device="/dev/vfio/0"'
        )
        return (
            "# HELP freetpu_exporter_up Whether exporter is up.\n"
            "# TYPE freetpu_exporter_up gauge\n"
            f"freetpu_exporter_up {self.exporter_up}\n"
            "# HELP freetpu_tpu_devices_discovered Total TPU devices.\n"
            "# TYPE freetpu_tpu_devices_discovered gauge\n"
            "freetpu_tpu_devices_discovered 1\n"
            "# HELP freetpu_tpu_health_status TPU readiness.\n"
            "# TYPE freetpu_tpu_health_status gauge\n"
            f"freetpu_tpu_health_status{{{labels}}} {self.health_status}\n"
            "# HELP freetpu_tpu_hbm_used_bytes HBM used.\n"
            "# TYPE freetpu_tpu_hbm_used_bytes gauge\n"
            f"freetpu_tpu_hbm_used_bytes{{{labels}}} {self.hbm_used_bytes}\n"
            "# HELP freetpu_tpu_hbm_total_bytes HBM total.\n"
            "# TYPE freetpu_tpu_hbm_total_bytes gauge\n"
            f"freetpu_tpu_hbm_total_bytes{{{labels}}} {self.hbm_total_bytes}\n"
            "# HELP freetpu_tpu_hbm_utilization_ratio HBM ratio.\n"
            "# TYPE freetpu_tpu_hbm_utilization_ratio gauge\n"
            f"freetpu_tpu_hbm_utilization_ratio{{{labels}}} {ratio:.6f}\n"
            "# HELP freetpu_tpu_duty_cycle_percent MXU duty cycle.\n"
            "# TYPE freetpu_tpu_duty_cycle_percent gauge\n"
            f"freetpu_tpu_duty_cycle_percent{{{labels}}} {self.duty_cycle_percent:.2f}\n"
            "# HELP freetpu_tpu_active_pids Active PIDs.\n"
            "# TYPE freetpu_tpu_active_pids gauge\n"
            f"freetpu_tpu_active_pids{{{labels}}} 1\n"
        )


class _MockTpuExporterHandler(BaseHTTPRequestHandler):
    server: _MockTpuExporter

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    def do_GET(self) -> None:  # noqa: N802
        body = self.server.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _RecordingScalerHook(ScalerHook):
    def __init__(self) -> None:
        self.pressure_events: List[QueuePressureEvent] = []
        self.idle_events: List[BackendIdleEvent] = []

    def on_queue_pressure(self, event: QueuePressureEvent) -> None:
        self.pressure_events.append(event)

    def on_backend_idle(self, event: BackendIdleEvent) -> None:
        self.idle_events.append(event)


def _start_server(srv: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return thread


def _http_request(
    port: int,
    method: str,
    path: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        req_headers = dict(headers or {})
        raw = None
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=raw, headers=req_headers)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


class TestFreeTPURouter(unittest.TestCase):
    def setUp(self) -> None:
        self._servers: List[ThreadingHTTPServer] = []
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        for srv in reversed(self._servers):
            if isinstance(srv, FreeTPURouterServer):
                srv.engine.stop_background_poller()
            srv.shutdown()
            srv.server_close()
        self._tmpdir.cleanup()

    def _track(self, srv: ThreadingHTTPServer) -> Any:
        srv.daemon_threads = True
        self._servers.append(srv)
        _start_server(srv)
        return srv

    def test_example_configs_validate(self) -> None:
        examples_dir = REPO_ROOT / "router" / "examples"
        files = sorted(examples_dir.glob("*.json"))
        self.assertGreaterEqual(len(files), 2)
        for path in files:
            cfg = load_config(path)
            self.assertGreater(len(cfg.backends), 0)

    def test_models_aggregation_streaming_and_usage_metrics(self) -> None:
        pixel = self._track(
            _MockOpenAIBackend(
                "pixel-g5-cluster",
                ["qwen2.5-1.5b-q8_0"],
                prompt_tokens=20,
                completion_tokens=10,
            )
        )
        tpu_exp = self._track(
            _MockTpuExporter(
                health_status=1,
                hbm_used_bytes=4 * 1024**3,
                hbm_total_bytes=16 * 1024**3,
                duty_cycle_percent=35.5,
                generation="v5e",
            )
        )
        cloud = self._track(
            _MockOpenAIBackend(
                "cloud-tpu-v5e",
                ["qwen2.5-1.5b-q8_0", "qwen2.5-32b-q4_0"],
                prompt_tokens=100,
                completion_tokens=50,
            )
        )

        jsonl_path = Path(self._tmpdir.name) / "usage.jsonl"
        captured_events: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(jsonl_path, callback=captured_events.append)

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="pixel-g5-cluster",
                    type="pixel-tpu",
                    local=True,
                    base_url=pixel.base_url,
                    cost_per_1m_prompt_usd=0.0,
                    cost_per_1m_completion_usd=0.0,
                ),
                BackendConfig(
                    id="cloud-tpu-v5e",
                    type="cloud-tpu-v5e",
                    local=False,
                    spot=True,
                    base_url=cloud.base_url,
                    tpu_exporter_url=tpu_exp.metrics_url,
                    cost_per_1m_prompt_usd=0.20,
                    cost_per_1m_completion_usd=0.40,
                ),
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        # 1. /v1/models aggregation
        status, _, body = _http_request(rport, "GET", "/v1/models")
        self.assertEqual(status, 200)
        models_json = json.loads(body)
        model_ids = [m["id"] for m in models_json["data"]]
        self.assertEqual(model_ids, ["qwen2.5-1.5b-q8_0", "qwen2.5-32b-q4_0"])

        # 2. /health includes TPU exporter HBM and duty cycle
        status, _, body = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 200)
        health_json = json.loads(body)
        self.assertEqual(health_json["healthy_backends"], 2)
        cloud_health = next(b for b in health_json["backends"] if b["id"] == "cloud-tpu-v5e")
        self.assertAlmostEqual(
            cloud_health["tpu_telemetry"]["max_hbm_utilization_ratio"], 0.25, places=3
        )
        self.assertAlmostEqual(
            cloud_health["tpu_telemetry"]["avg_duty_cycle_percent"], 35.5, places=1
        )

        # 3. Non-streaming chat completion -> local_first picks pixel-g5-cluster
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "qwen2.5-1.5b-q8_0",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "pixel-g5-cluster")
        self.assertIn("reply-from:pixel-g5-cluster", json.loads(body)["choices"][0]["message"]["content"])

        # 4. SSE streaming pass-through -> routes 32B model to cloud-tpu-v5e
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "qwen2.5-32b-q4_0",
                "stream": True,
                "messages": [{"role": "user", "content": "stream test"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "cloud-tpu-v5e")
        self.assertIn("text/event-stream", hdrs["content-type"])
        sse_text = body.decode("utf-8")
        self.assertIn("data: [DONE]", sse_text)
        self.assertIn("cloud-tpu-v5e", sse_text)

        # 5. /v1/completions endpoint
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={"model": "qwen2.5-1.5b-q8_0", "prompt": "Hello"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["text"], "completion-from:pixel-g5-cluster")

        # 6. UsageMeterHook callback + JSONL accounting
        self.assertEqual(len(captured_events), 3)
        cloud_ev = captured_events[1]
        self.assertEqual(cloud_ev.backend_id, "cloud-tpu-v5e")
        self.assertEqual(cloud_ev.prompt_tokens, 100)
        self.assertEqual(cloud_ev.completion_tokens, 50)
        # Cost: 100 * 0.20 / 1e6 + 50 * 0.40 / 1e6 = 0.00004 USD
        self.assertAlmostEqual(cloud_ev.estimated_cost_usd, 0.00004, places=8)
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)

        # 7. Prometheus /metrics exposition 0.0.4
        status, hdrs, body = _http_request(rport, "GET", "/metrics")
        self.assertEqual(status, 200)
        metrics_txt = body.decode("utf-8")
        self.assertIn("freetpu_router_requests_total", metrics_txt)
        self.assertIn("freetpu_router_active_inflight", metrics_txt)
        self.assertIn("freetpu_router_failovers_total", metrics_txt)
        self.assertIn("freetpu_router_prompt_tokens_total", metrics_txt)
        self.assertIn("freetpu_router_completion_tokens_total", metrics_txt)
        self.assertIn("freetpu_router_ttft_seconds_bucket", metrics_txt)
        self.assertIn("freetpu_router_estimated_cost_usd_total", metrics_txt)

    def test_spot_tpu_preemption_503_failover_json_and_stream(self) -> None:
        # Healthy at startup probe so it is selected first, then preempted with 503.
        spot_tpu = self._track(
            _MockOpenAIBackend("spot-v5e-1", ["qwen2.5-32b-q4_0"], status_code=200)
        )
        backup_tpu = self._track(
            _MockOpenAIBackend("ondemand-v6e-1", ["qwen2.5-32b-q4_0"], status_code=200)
        )

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-v5e-1",
                    type="cloud-tpu-v5e",
                    local=False,
                    spot=True,
                    base_url=spot_tpu.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="ondemand-v6e-1",
                    type="cloud-tpu-v6e",
                    local=False,
                    spot=False,
                    base_url=backup_tpu.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Simulate Spot preemption (HTTP 503) before non-streaming request.
        spot_tpu.status_code = 503
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "qwen2.5-32b-q4_0",
                "messages": [{"role": "user", "content": "survive preemption"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "ondemand-v6e-1")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:ondemand-v6e-1", json.loads(body)["choices"][0]["message"]["content"])

        # Now reset cooldown manually and simulate pre-first-token stream abort on spot_tpu.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        spot_tpu.status_code = 200
        spot_tpu.abort_stream_before_first_token = True

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "qwen2.5-32b-q4_0",
                "stream": True,
                "messages": [{"role": "user", "content": "survive stream abort"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "ondemand-v6e-1")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("ondemand-v6e-1", body.decode("utf-8"))
        self.assertIn("data: [DONE]", body.decode("utf-8"))

        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_failovers_total{model="qwen2.5-32b-q4_0",from_backend="spot-v5e-1",to_backend="ondemand-v6e-1",reason="http_503"} 1',
            metrics_txt,
        )
        self.assertIn(
            'freetpu_router_failovers_total{model="qwen2.5-32b-q4_0",from_backend="spot-v5e-1",to_backend="ondemand-v6e-1",reason="stream_aborted_before_first_token"} 1',
            metrics_txt,
        )

    def test_lowest_latency_and_connection_error_failover(self) -> None:
        slow_backend = self._track(
            _MockOpenAIBackend("slow-tpu", ["m-lat"], latency_s=0.04)
        )
        fast_backend = self._track(
            _MockOpenAIBackend("fast-tpu", ["m-lat"], latency_s=0.0)
        )

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="lowest_latency",
            backends=[
                BackendConfig(
                    id="slow-tpu",
                    type="cloud-tpu-v5e",
                    base_url=slow_backend.base_url,
                    latency_hint_ms=80.0,
                ),
                BackendConfig(
                    id="fast-tpu",
                    type="cloud-tpu-v6e",
                    base_url=fast_backend.base_url,
                    latency_hint_ms=5.0,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Picks fast-tpu via lowest_latency
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-lat", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "fast-tpu")

        # Simulate sudden socket death on fast-tpu -> seamlessly fails over to slow-tpu
        fast_backend.shutdown()
        fast_backend.server_close()
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-lat", "messages": [{"role": "user", "content": "hi2"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "slow-tpu")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:slow-tpu", json.loads(body)["choices"][0]["message"]["content"])

    def test_cheapest_vs_local_first_vs_round_robin_policies(self) -> None:
        local_slow_paid = self._track(
            _MockOpenAIBackend("local-node", ["shared-model"], latency_s=0.03)
        )
        remote_cheap_fast = self._track(
            _MockOpenAIBackend("remote-cheap", ["shared-model"], latency_s=0.0)
        )

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            backends=[
                BackendConfig(
                    id="local-node",
                    type="pixel-tpu",
                    local=True,
                    base_url=local_slow_paid.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                    latency_hint_ms=80.0,
                ),
                BackendConfig(
                    id="remote-cheap",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=remote_cheap_fast.base_url,
                    cost_per_1m_prompt_usd=0.05,
                    cost_per_1m_completion_usd=0.05,
                    latency_hint_ms=10.0,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Default policy = cheapest -> remote-cheap
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "shared-model", "messages": [{"role": "user", "content": "a"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "remote-cheap")

        # Per-request header override -> local_first -> local-node
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"X-FreeTPU-Policy": "local_first"},
            body={"model": "shared-model", "messages": [{"role": "user", "content": "b"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "local-node")

        # Round-robin alternates between backends
        seen_rr: List[str] = []
        for _ in range(2):
            status, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                headers={"X-FreeTPU-Policy": "round_robin"},
                body={"model": "shared-model", "messages": [{"role": "user", "content": "rr"}]},
            )
            self.assertEqual(status, 200)
            seen_rr.append(hdrs["x-freetpu-backend"])
        self.assertEqual(set(seen_rr), {"local-node", "remote-cheap"})

    def test_tpu_exporter_avoids_oom_and_unhealthy_tpu(self) -> None:
        exp_overloaded = self._track(
            _MockTpuExporter(
                health_status=1,
                hbm_used_bytes=15 * 1024**3,
                hbm_total_bytes=16 * 1024**3,  # 0.9375 >= 0.90 threshold
                duty_cycle_percent=88.0,
            )
        )
        exp_healthy = self._track(
            _MockTpuExporter(
                health_status=1,
                hbm_used_bytes=2 * 1024**3,
                hbm_total_bytes=16 * 1024**3,
                duty_cycle_percent=12.0,
            )
        )
        tpu_a = self._track(_MockOpenAIBackend("tpu-a", ["qwen-72b"]))
        tpu_b = self._track(_MockOpenAIBackend("tpu-b", ["qwen-72b"]))

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            backends=[
                BackendConfig(
                    id="tpu-a",
                    type="cloud-tpu-v5e",
                    base_url=tpu_a.base_url,
                    tpu_exporter_url=exp_overloaded.metrics_url,
                    max_hbm_utilization=0.90,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="tpu-b",
                    type="cloud-tpu-v6e",
                    base_url=tpu_b.base_url,
                    tpu_exporter_url=exp_healthy.metrics_url,
                    max_hbm_utilization=0.90,
                    cost_per_1m_prompt_usd=0.40,
                    cost_per_1m_completion_usd=0.40,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Even though tpu-a is cheaper, HBM=93.75% >= 90% avoids OOM and picks tpu-b.
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "qwen-72b", "messages": [{"role": "user", "content": "check"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "tpu-b")

        # Verify /health reflects tpu_hbm_oom_risk on tpu-a
        _, _, hbody = _http_request(rport, "GET", "/health")
        hjson = json.loads(hbody)
        tpu_a_snap = next(b for b in hjson["backends"] if b["id"] == "tpu-a")
        self.assertFalse(tpu_a_snap["healthy"])
        self.assertEqual(tpu_a_snap["unhealthy_reason"], "tpu_hbm_oom_risk")

        # Now relieve HBM on tpu-a, but flip tpu-b health_status=0 -> routes back to tpu-a
        exp_overloaded.hbm_used_bytes = 4 * 1024**3
        exp_healthy.health_status = 0
        router.engine.refresh_all_backends()

        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "qwen-72b", "messages": [{"role": "user", "content": "recovered"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "tpu-a")

    def test_auth_hook_and_scaler_hook_lifecycle(self) -> None:
        backend = self._track(_MockOpenAIBackend("pixel-1", ["m1"]))
        scaler = _RecordingScalerHook()
        auth = BearerTokenAuthHook({"secret-dev-token": "dev-user"})

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backend_idle_timeout_s=0.01,
            backends=[
                BackendConfig(
                    id="pixel-1",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, auth_hook=auth, scaler_hook=scaler))
        rport = router.server_address[1]

        # /health is exempt by default
        status, _, _ = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 200)

        # /v1/models without token -> 401
        status, _, _ = _http_request(rport, "GET", "/v1/models")
        self.assertEqual(status, 401)

        # Unknown model with valid token -> 503 + ScalerHook.on_queue_pressure
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret-dev-token"},
            body={"model": "missing-model", "messages": [{"role": "user", "content": "x"}]},
        )
        self.assertEqual(status, 503)
        self.assertEqual(len(scaler.pressure_events), 1)
        self.assertEqual(scaler.pressure_events[0].reason, "no_healthy_backend")

        # Idle callback fires once backend exceeds idle threshold
        time.sleep(0.02)
        router.engine.check_idle_backends()
        self.assertEqual(len(scaler.idle_events), 1)
        self.assertEqual(scaler.idle_events[0].backend_id, "pixel-1")

    def test_chunked_sse_stream_with_keepalive_upstream_and_split_event(self) -> None:
        hold_open = threading.Event()
        backend = self._track(
            _MockOpenAIBackend(
                "llama-chunked",
                ["qwen-chunked"],
                chunked_keepalive_sse=True,
                keepalive_hold_event=hold_open,
                prompt_tokens=15,
                completion_tokens=7,
            )
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "chunked.jsonl", callback=captured.append
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="llama-chunked",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        t0 = time.monotonic()
        try:
            status, hdrs, body = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "qwen-chunked",
                    "stream": True,
                    "messages": [{"role": "user", "content": "chunked test"}],
                },
            )
        finally:
            hold_open.set()
        elapsed = time.monotonic() - t0

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", hdrs["content-type"])
        # Must finish right after terminating 0-chunk instead of waiting for keep-alive hold (3s).
        self.assertLess(elapsed, 1.5)
        sse_text = body.decode("utf-8")
        self.assertNotIn("\r\n0\r\n", sse_text)
        # Every non-empty line in the proxied body must be a valid SSE data line (no hex chunk sizes).
        data_payloads: List[str] = []
        for raw_line in sse_text.splitlines():
            if not raw_line:
                continue
            self.assertTrue(
                raw_line.startswith("data: "),
                f"Unexpected non-SSE line (leaked chunk framing?): {raw_line!r}",
            )
            data_payloads.append(raw_line[6:])
        self.assertEqual(data_payloads[-1], "[DONE]")
        parsed = [json.loads(p) for p in data_payloads[:-1]]
        combined = "".join(p["choices"][0]["delta"].get("content", "") for p in parsed)
        self.assertEqual(combined, "split-ok")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].prompt_tokens, 15)
        self.assertEqual(captured[0].completion_tokens, 7)

    def test_unknown_auth_mode_rejected_by_config_and_check_config(self) -> None:
        base_backend = BackendConfig(
            id="b1",
            type="pixel-tpu",
            base_url="http://127.0.0.1:8081/v1",
        )
        with self.assertRaises(ValueError):
            RouterConfig(backends=[base_backend], auth={"type": "beaer"})

        bad_cfg_path = Path(self._tmpdir.name) / "bad_auth.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "auth": {"type": "bearer_typo", "tokens": ["secret"]},
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_config(bad_cfg_path)

        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("Invalid auth.type", stderr_buf.getvalue())

        # Direct mutation before hook construction must also fail closed.
        valid_cfg = RouterConfig(backends=[base_backend], auth={"type": "allow_all"})
        valid_cfg.auth = {"type": "unknown_mode"}
        with self.assertRaises(ValueError):
            build_hooks_from_config(valid_cfg)

    def test_cors_origin_restriction_and_explicit_opt_in(self) -> None:
        backend = self._track(_MockOpenAIBackend("pixel-cors", ["m-cors"]))
        default_cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="pixel-cors",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router_default = self._track(create_router_server(default_cfg))
        rport = router_default.server_address[1]

        # Default: OPTIONS from untrusted browser origin is rejected without ACAO.
        status, hdrs, _ = _http_request(
            rport,
            "OPTIONS",
            "/v1/chat/completions",
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(status, 403)
        self.assertNotIn("access-control-allow-origin", hdrs)

        # Default: untrusted Origin on POST (both JSON and text/plain simple request) -> 403.
        for ctype in ("application/json", "text/plain"):
            status, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                headers={"Origin": "https://evil.example", "Content-Type": ctype},
                body={"model": "m-cors", "messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(status, 403)
            self.assertNotIn("access-control-allow-origin", hdrs)

        # Null origin (sandboxed iframe / local page) also rejected.
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "null"},
            body={"model": "m-cors", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 403)
        self.assertEqual(len(backend.requests_seen), 0)

        # Non-browser client without Origin succeeds and receives no wildcard ACAO.
        status, hdrs, _ = _http_request(rport, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertNotIn("access-control-allow-origin", hdrs)

        # Explicit opt-in allows only configured origin.
        optin_cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            cors_allowed_origins=["https://trusted.example"],
            backends=[
                BackendConfig(
                    id="pixel-cors",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router_optin = self._track(create_router_server(optin_cfg))
        oport = router_optin.server_address[1]

        status, hdrs, _ = _http_request(
            oport,
            "OPTIONS",
            "/v1/chat/completions",
            headers={"Origin": "https://trusted.example"},
        )
        self.assertEqual(status, 204)
        self.assertEqual(hdrs.get("access-control-allow-origin"), "https://trusted.example")
        self.assertEqual(hdrs.get("vary"), "Origin")

        status, hdrs, _ = _http_request(
            oport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://trusted.example"},
            body={
                "model": "m-cors",
                "stream": True,
                "messages": [{"role": "user", "content": "ok"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs.get("access-control-allow-origin"), "https://trusted.example")
        self.assertEqual(hdrs.get("vary"), "Origin")

        status, hdrs, _ = _http_request(
            oport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://evil.example"},
            body={"model": "m-cors", "messages": [{"role": "user", "content": "no"}]},
        )
        self.assertEqual(status, 403)
        self.assertNotIn("access-control-allow-origin", hdrs)

    def test_inbound_request_size_limit_and_read_deadline(self) -> None:
        backend = self._track(_MockOpenAIBackend("pixel-bound", ["m-bound"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            max_request_body_bytes=128,
            client_read_timeout_s=0.25,
            backends=[
                BackendConfig(
                    id="pixel-bound",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Oversized Content-Length -> 413 Payload Too Large without touching backend.
        status, _, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-bound",
                "messages": [{"role": "user", "content": "x" * 256}],
            },
        )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"]["code"], "payload_too_large")
        self.assertEqual(len(backend.requests_seen), 0)

        # 2. Stalled unfinished upload hits client_read_timeout_s and returns 408 promptly.
        t0 = time.monotonic()
        with socket.create_connection(("127.0.0.1", rport), timeout=2.0) as sock:
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
                b'{"mod'
            )
            sock.settimeout(2.0)
            resp_raw = sock.recv(4096)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 1.5)
        self.assertIn(b"408", resp_raw.splitlines()[0])
        self.assertEqual(len(backend.requests_seen), 0)

    def test_client_disconnect_does_not_failover_or_cooldown_backend(self) -> None:
        primary = self._track(
            _MockOpenAIBackend("slow-primary", ["m-disc"], latency_s=0.25)
        )
        fallback = self._track(
            _MockOpenAIBackend("paid-fallback", ["m-disc"], latency_s=0.0)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="slow-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="paid-fallback",
                    type="cloud-tpu-v6e",
                    local=False,
                    base_url=fallback.base_url,
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=2.0,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        payload = json.dumps(
            {"model": "m-disc", "messages": [{"role": "user", "content": "cancel me"}]}
        ).encode("utf-8")
        req_bytes = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
            + payload
        )
        sock = socket.create_connection(("127.0.0.1", rport), timeout=2.0)
        sock.sendall(req_bytes)
        # Wait briefly so primary begins processing, then close client socket abruptly.
        time.sleep(0.06)
        sock.close()

        # Wait for primary to finish replying and handler to complete.
        time.sleep(0.35)
        primary_state = router.engine._backends[0]
        self.assertEqual(len(primary.requests_seen), 1)
        self.assertEqual(len(fallback.requests_seen), 0)
        self.assertEqual(primary_state.total_failures, 0)
        self.assertEqual(primary_state.cooldown_until_monotonic, 0.0)
        self.assertTrue(primary_state.healthy)

    def test_slot_recheck_prevents_overloading_fallback_and_respects_max_inflight(self) -> None:
        primary = self._track(
            _MockOpenAIBackend("slow-fail-primary", ["m-cap"], status_code=200, latency_s=0.0)
        )
        fallback = self._track(
            _MockOpenAIBackend("cap-fallback", ["m-cap"], status_code=200, latency_s=0.30)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="slow-fail-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="cap-fallback",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=fallback.base_url,
                    max_inflight=1,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Primary will now fail with 503 after 0.15s.
        primary.status_code = 503
        primary.latency_s = 0.15

        results: List[Tuple[int, Dict[str, str]]] = []

        def run_req() -> None:
            st, hd, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": "m-cap", "messages": [{"role": "user", "content": "race"}]},
            )
            results.append((st, hd))

        # Request 1 starts, hits primary (takes 0.15s to fail).
        t1 = threading.Thread(target=run_req)
        t1.start()
        time.sleep(0.05)

        # While Request 1 is waiting on primary, occupy fallback's single slot.
        fallback_state = router.engine._backends[1]
        reserved = router.engine.try_reserve_backend(fallback_state)
        self.assertIsNotNone(reserved)
        try:
            t1.join(timeout=3.0)
        finally:
            router.engine.release_backend(fallback_state)

        # Fallback was full when Request 1 tried to fail over -> must NOT overload fallback.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 503)
        self.assertEqual(len(fallback.requests_seen), 0)
        self.assertEqual(fallback.peak_inflight, 0)

    def test_sse_keepalive_before_eof_fails_over_to_healthy_fallback(self) -> None:
        spot = self._track(
            _MockOpenAIBackend("spot-keepalive", ["m-sse"], status_code=200)
        )
        healthy = self._track(
            _MockOpenAIBackend("healthy-backup", ["m-sse"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-keepalive",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=spot.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="healthy-backup",
                    type="cloud-tpu-v6e",
                    spot=False,
                    base_url=healthy.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Case 1: Spot sends SSE keepalive comment + blank line, then EOF (preempted).
        spot.raw_sse_bytes = b": keep-alive\n\n"
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse",
                "stream": True,
                "messages": [{"role": "user", "content": "survive keepalive eof"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("healthy-backup", body.decode("utf-8"))

        # Case 2: Spot sends incomplete data line without terminating blank line before EOF.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        spot.raw_sse_bytes = b'data: {"id":"partial","choices":[]}\n'
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse",
                "stream": True,
                "messages": [{"role": "user", "content": "survive truncated event"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("healthy-backup", body.decode("utf-8"))

    def test_streamed_tool_calls_and_multi_choice_usage_accounting(self) -> None:
        tool_chunks = [
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "model": "m-tool",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_tpu_status",
                                        "arguments": '{"zone": "us-central1-a", ',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "model": "m-tool",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"chip_count": 8}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        backend = self._track(
            _MockOpenAIBackend(
                "tool-worker",
                ["m-tool"],
                custom_sse_chunks=tool_chunks,
            )
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "tool_usage.jsonl", callback=captured.append
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="tool-worker",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=backend.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=1.00,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-tool",
                "stream": True,
                "messages": [{"role": "user", "content": "check tpu"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        self.assertGreater(captured[0].completion_tokens, 0)
        self.assertGreater(captured[0].estimated_cost_usd, 0.0)
        self.assertIsNotNone(captured[0].ttft_ms)

        # Multi-choice stream where only choice[1] emits text and upstream omits usage.
        backend.custom_sse_chunks = [
            {
                "id": "chatcmpl-multi",
                "object": "chat.completion.chunk",
                "model": "m-tool",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": None},
                    {"index": 1, "delta": {"content": "second choice output text"}, "finish_reason": "stop"},
                ],
            }
        ]
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-tool",
                "stream": True,
                "n": 2,
                "messages": [{"role": "user", "content": "multi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 2)
        self.assertGreater(captured[1].completion_tokens, 0)

    def test_tpu_exporter_rejects_html_and_unrelated_metrics(self) -> None:
        html_tele = summarize_tpu_samples(
            parse_prometheus_text("<html><body><h1>Hello</h1></body></html>")
        )
        self.assertFalse(html_tele.exporter_up)
        ready, reason = evaluate_tpu_readiness(html_tele)
        self.assertFalse(ready)
        self.assertEqual(reason, "tpu_exporter_invalid_metrics")

        unrelated_tele = summarize_tpu_samples(
            parse_prometheus_text(
                "# TYPE node_cpu_seconds_total counter\nnode_cpu_seconds_total 42.0\n"
            )
        )
        self.assertFalse(unrelated_tele.exporter_up)
        ready, reason = evaluate_tpu_readiness(unrelated_tele)
        self.assertFalse(ready)
        self.assertEqual(reason, "tpu_exporter_invalid_metrics")

    def test_disabled_backend_excluded_from_health_readiness(self) -> None:
        down_enabled = self._track(
            _MockOpenAIBackend("enabled-down", ["m-h"], status_code=503)
        )
        spare = self._track(
            _MockOpenAIBackend("disabled-spare", ["m-h"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="enabled-down",
                    type="pixel-tpu",
                    local=True,
                    base_url=down_enabled.base_url,
                    enabled=True,
                ),
                BackendConfig(
                    id="disabled-spare",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=spare.base_url,
                    enabled=False,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, _, body = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 503)
        hjson = json.loads(body)
        self.assertEqual(hjson["status"], "degraded")
        self.assertEqual(hjson["healthy_backends"], 0)
        self.assertEqual(hjson["enabled_backends"], 1)
        spare_snap = next(b for b in hjson["backends"] if b["id"] == "disabled-spare")
        self.assertFalse(spare_snap["healthy"])
        self.assertFalse(spare_snap["enabled"])

    def test_idle_timer_starts_when_inflight_drops_to_zero(self) -> None:
        # Request lasts 0.14s, longer than backend_idle_timeout_s=0.06s.
        slow_worker = self._track(
            _MockOpenAIBackend("long-worker", ["m-idle"], latency_s=0.14)
        )
        scaler = _RecordingScalerHook()
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backend_idle_timeout_s=0.06,
            backends=[
                BackendConfig(
                    id="long-worker",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=slow_worker.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, scaler_hook=scaler))
        rport = router.server_address[1]

        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-idle", "messages": [{"role": "user", "content": "slow"}]},
        )
        self.assertEqual(status, 200)
        # Right after request completion, idle time should be ~0s < 0.06s -> no idle event.
        router.engine.check_idle_backends()
        self.assertEqual(len(scaler.idle_events), 0)

        # After waiting > 0.06s post-completion, idle event fires.
        time.sleep(0.08)
        router.engine.check_idle_backends()
        self.assertEqual(len(scaler.idle_events), 1)
        self.assertEqual(scaler.idle_events[0].backend_id, "long-worker")

    def test_jsonl_usage_hook_does_not_retain_events_in_memory(self) -> None:
        jsonl_path = Path(self._tmpdir.name) / "bounded.jsonl"
        hook = JsonlUsageMeterHook(jsonl_path)
        for idx in range(25):
            hook.record_usage(
                UsageEvent(
                    request_id=f"req-{idx}",
                    model="m1",
                    backend_id="b1",
                    prompt_tokens=10,
                    completion_tokens=5,
                    latency_ms=12.5,
                    estimated_cost_usd=0.0,
                )
            )
        self.assertEqual(hook.events_written, 25)
        self.assertFalse(hasattr(hook, "events"))
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 25)

    def test_terminal_routing_failures_counted_in_request_metrics(self) -> None:
        failing_503 = self._track(
            _MockOpenAIBackend("worker-503", ["m-503"], status_code=200)
        )
        dying_conn = self._track(
            _MockOpenAIBackend("worker-dead", ["m-dead"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="worker-503",
                    type="cloud-tpu-v5e",
                    base_url=failing_503.base_url,
                ),
                BackendConfig(
                    id="worker-dead",
                    type="cloud-tpu-v5e",
                    base_url=dying_conn.base_url,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. No eligible backend for unknown model -> 503 counted under backend_id="none".
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "unknown-model", "messages": [{"role": "user", "content": "x"}]},
        )
        self.assertEqual(status, 503)

        # 2. Terminal retryable 503 on sole candidate -> counted under worker-503, status="503".
        failing_503.status_code = 503
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-503", "messages": [{"role": "user", "content": "y"}]},
        )
        self.assertEqual(status, 503)

        # 3. Terminal connection error on sole candidate -> counted under worker-dead, status="502".
        dying_conn.shutdown()
        dying_conn.server_close()
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-dead", "messages": [{"role": "user", "content": "z"}]},
        )
        self.assertEqual(status, 502)

        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="_unknown",backend_id="none",status="503"} 1',
            metrics_txt,
        )
        self.assertNotIn('model="unknown-model"', metrics_txt)
        self.assertIn(
            'freetpu_router_requests_total{model="m-503",backend_id="worker-503",status="503"} 1',
            metrics_txt,
        )
        self.assertIn(
            'freetpu_router_requests_total{model="m-dead",backend_id="worker-dead",status="502"} 1',
            metrics_txt,
        )

    def test_metrics_model_label_bounded_for_unknown_and_wildcard_models(self) -> None:
        known_worker = self._track(_MockOpenAIBackend("known-worker", ["known-model"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="known-worker",
                    type="pixel-tpu",
                    local=True,
                    base_url=known_worker.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 20 unique unknown model names must collapse into one bounded "_unknown" series.
        for idx in range(20):
            status, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": f"attacker-random-model-{idx}",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(status, 503)

        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="_unknown",backend_id="none",status="503"} 20',
            metrics_txt,
        )
        self.assertNotIn("attacker-random-model-", metrics_txt)

        # Wildcard backend: arbitrary client model names must collapse to "_other".
        wc_worker = self._track(_MockOpenAIBackend("wildcard-worker", []))
        wc_cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="round_robin",
            backends=[
                BackendConfig(
                    id="wildcard-worker",
                    type="cloud-tpu-v5e",
                    base_url=wc_worker.base_url,
                    models=["*"],
                    discover_models=False,
                )
            ],
        )
        wc_router = self._track(create_router_server(wc_cfg))
        wport = wc_router.server_address[1]
        for idx in range(20):
            status, _, _ = _http_request(
                wport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": f"wildcard-arbitrary-{idx}",
                    "messages": [{"role": "user", "content": "ok"}],
                },
            )
            self.assertEqual(status, 200)

        _, _, wc_metrics_raw = _http_request(wport, "GET", "/metrics")
        wc_metrics_txt = wc_metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="_other",backend_id="wildcard-worker",status="200"} 20',
            wc_metrics_txt,
        )
        self.assertNotIn("wildcard-arbitrary-", wc_metrics_txt)
        self.assertLessEqual(len(wc_router.engine._rr_cursors), MAX_RR_CURSORS)

    def test_scalar_bearer_tokens_and_exempt_paths_rejected(self) -> None:
        base_backend = BackendConfig(
            id="b1",
            type="pixel-tpu",
            base_url="http://127.0.0.1:8081/v1",
        )
        with self.assertRaises(ValueError):
            RouterConfig(
                backends=[base_backend],
                auth={"type": "bearer", "tokens": "secret-scalar"},
            )
        with self.assertRaises(ValueError):
            RouterConfig(
                backends=[base_backend],
                auth={"type": "bearer", "tokens": ["valid"], "tokens_env": "MY_TOKEN_ENV"},
            )
        with self.assertRaises(ValueError):
            RouterConfig(
                backends=[base_backend],
                auth={"type": "bearer", "tokens": ["valid"], "exempt_paths": "/health"},
            )
        with self.assertRaises(ValueError):
            BearerTokenAuthHook("secret-scalar")

        bad_cfg_path = Path(self._tmpdir.name) / "scalar_tokens.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "auth": {"type": "bearer", "tokens": "secret-scalar"},
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("auth.tokens", stderr_buf.getvalue())

        # Direct mutation before build_hooks_from_config must also reject scalar tokens.
        cfg = RouterConfig(
            backends=[base_backend],
            auth={"type": "bearer", "tokens": ["valid-token"]},
        )
        cfg.auth = {"type": "bearer", "tokens": "scalar-after-init"}
        with self.assertRaises(ValueError):
            build_hooks_from_config(cfg)

    def test_sse_role_only_event_before_eof_fails_over_to_healthy_fallback(self) -> None:
        spot = self._track(_MockOpenAIBackend("spot-role-only", ["m-role"]))
        healthy = self._track(_MockOpenAIBackend("healthy-role-backup", ["m-role"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-role-only",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=spot.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="healthy-role-backup",
                    type="cloud-tpu-v6e",
                    spot=False,
                    base_url=healthy.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Spot sends a complete role-only SSE event and is immediately preempted (EOF).
        role_chunk = json.dumps(
            {
                "id": "chatcmpl-spot",
                "object": "chat.completion.chunk",
                "model": "m-role",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
        )
        spot.raw_sse_bytes = f"data: {role_chunk}\n\n".encode("utf-8")

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-role",
                "stream": True,
                "messages": [{"role": "user", "content": "survive role-only eof"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-role-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        sse_text = body.decode("utf-8")
        self.assertIn("healthy-role-backup", sse_text)
        self.assertIn("data: [DONE]", sse_text)

        # Verify role-only event followed by a valid finish_reason terminator commits cleanly
        # and preserves the buffered role-only event in exact order.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        stop_chunk = json.dumps(
            {
                "id": "chatcmpl-spot",
                "object": "chat.completion.chunk",
                "model": "m-role",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        spot.raw_sse_bytes = (
            f"data: {role_chunk}\n\ndata: {stop_chunk}\n\ndata: [DONE]\n\n"
        ).encode("utf-8")

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-role",
                "stream": True,
                "messages": [{"role": "user", "content": "empty valid stop"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "spot-role-only")
        self.assertEqual(hdrs["x-freetpu-failovers"], "0")
        sse_text = body.decode("utf-8")
        self.assertIn('"role": "assistant"', sse_text)
        self.assertIn('"finish_reason": "stop"', sse_text)

    def test_create_router_server_runs_periodic_health_polling_in_serve_forever(self) -> None:
        backend = self._track(
            _MockOpenAIBackend("recovering-worker", ["m-poll"], status_code=503)
        )
        scaler = _RecordingScalerHook()
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            health_poll_interval_s=0.05,
            backend_idle_timeout_s=0.04,
            backends=[
                BackendConfig(
                    id="recovering-worker",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=backend.base_url,
                )
            ],
        )
        # Use create_router_server() + serve_forever() directly without calling
        # start_background_poller() manually.
        router = self._track(create_router_server(cfg, scaler_hook=scaler))
        rport = router.server_address[1]

        status, _, _ = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 503)

        # Recover the backend; background poller started by serve_forever() must detect it
        # and also emit idle scaler events automatically.
        backend.status_code = 200
        deadline = time.monotonic() + 1.5
        recovered = False
        while time.monotonic() < deadline:
            st, _, _ = _http_request(rport, "GET", "/health")
            if st == 200 and len(scaler.idle_events) >= 1:
                recovered = True
                break
            time.sleep(0.05)

        self.assertTrue(recovered, "Background poller did not refresh backend or idle hooks")
        self.assertEqual(scaler.idle_events[0].backend_id, "recovering-worker")

    def test_non_finite_tpu_telemetry_contained_to_affected_backend(self) -> None:
        bad_exporter = self._track(
            _MockTpuExporter(
                raw_body=(
                    "freetpu_exporter_up 1\n"
                    "freetpu_tpu_devices_discovered NaN\n"
                    'freetpu_tpu_health_status{chip="0"} +Inf\n'
                    'freetpu_tpu_hbm_used_bytes{chip="0"} +Inf\n'
                    'freetpu_tpu_hbm_total_bytes{chip="0"} 17179869184\n'
                    'freetpu_tpu_hbm_utilization_ratio{chip="0"} NaN\n'
                    'freetpu_tpu_active_pids{chip="0"} -Inf\n'
                )
            )
        )
        good_exporter = self._track(
            _MockTpuExporter(
                health_status=1,
                hbm_used_bytes=2 * 1024**3,
                hbm_total_bytes=16 * 1024**3,
                duty_cycle_percent=15.0,
            )
        )
        bad_tpu = self._track(_MockOpenAIBackend("tpu-bad-tele", ["m-nan"]))
        good_tpu = self._track(_MockOpenAIBackend("tpu-good-tele", ["m-nan"]))

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            backends=[
                BackendConfig(
                    id="tpu-bad-tele",
                    type="cloud-tpu-v5e",
                    base_url=bad_tpu.base_url,
                    tpu_exporter_url=bad_exporter.metrics_url,
                    cost_per_1m_prompt_usd=0.05,
                    cost_per_1m_completion_usd=0.05,
                ),
                BackendConfig(
                    id="tpu-good-tele",
                    type="cloud-tpu-v6e",
                    base_url=good_tpu.base_url,
                    tpu_exporter_url=good_exporter.metrics_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        # Startup initial probe must not raise and must still probe tpu-good-tele.
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, _, hbody = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 200)
        hjson = json.loads(hbody)
        self.assertEqual(hjson["healthy_backends"], 1)
        bad_snap = next(b for b in hjson["backends"] if b["id"] == "tpu-bad-tele")
        good_snap = next(b for b in hjson["backends"] if b["id"] == "tpu-good-tele")
        self.assertFalse(bad_snap["healthy"])
        self.assertEqual(bad_snap["unhealthy_reason"], "tpu_exporter_invalid_metrics")
        self.assertTrue(good_snap["healthy"])

        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-nan", "messages": [{"role": "user", "content": "route around NaN"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "tpu-good-tele")

    def test_bearer_auth_honors_explicit_empty_exempt_paths(self) -> None:
        backend = self._track(_MockOpenAIBackend("auth-strict", ["m-strict"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            auth={
                "type": "bearer",
                "tokens": ["strict-secret-token"],
                "exempt_paths": [],
            },
            backends=[
                BackendConfig(
                    id="auth-strict",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # With exempt_paths=[], even /health must require bearer auth.
        status, _, _ = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 401)

        status, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Authorization": "Bearer strict-secret-token"},
        )
        self.assertEqual(status, 200)

    def test_bearer_credentials_without_bearer_mode_rejected(self) -> None:
        base_backend = BackendConfig(
            id="b1",
            type="pixel-tpu",
            base_url="http://127.0.0.1:8081/v1",
        )
        with self.assertRaises(ValueError):
            RouterConfig(backends=[base_backend], auth={"tokens": ["secret"]})
        with self.assertRaises(ValueError):
            RouterConfig(
                backends=[base_backend],
                auth={"type": "allow_all", "tokens_env": ["MY_TOKEN"]},
            )
        with self.assertRaises(ValueError):
            RouterConfig(backends=[base_backend], auth={"exempt_paths": []})

        bad_cfg_path = Path(self._tmpdir.name) / "missing_bearer_type.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "auth": {"tokens": ["secret-token"]},
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("auth.type='bearer'", stderr_buf.getvalue())

        cfg = RouterConfig(backends=[base_backend], auth={"type": "allow_all"})
        cfg.auth = {"tokens": ["secret-after-init"]}
        with self.assertRaises(ValueError):
            build_hooks_from_config(cfg)

    def test_connection_cap_and_slowloris_header_deadline(self) -> None:
        backend = self._track(_MockOpenAIBackend("conn-worker", ["m-conn"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            max_concurrent_connections=2,
            client_header_timeout_s=0.30,
            backends=[
                BackendConfig(
                    id="conn-worker",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Occupy both connection slots with unfinished header requests.
        held_socks = [
            socket.create_connection(("127.0.0.1", rport), timeout=2.0),
            socket.create_connection(("127.0.0.1", rport), timeout=2.0),
        ]
        try:
            for s in held_socks:
                s.sendall(b"GET ")
            time.sleep(0.05)

            # 3rd concurrent connection must be rejected immediately with 503.
            status, _, body = _http_request(rport, "GET", "/health")
            self.assertEqual(status, 503)
            self.assertEqual(json.loads(body)["error"]["code"], "server_overloaded")
        finally:
            for s in held_socks:
                s.close()

        # Wait briefly for the 2 closed connections to release their slots.
        time.sleep(0.05)

        # Slowloris trickle headers (1 header every 0.06s) must hit the total 0.30s header deadline.
        t0 = time.monotonic()
        with socket.create_connection(("127.0.0.1", rport), timeout=2.0) as slow_sock:
            slow_sock.sendall(b"GET /health HTTP/1.1\r\n")
            closed_by_server = False
            while time.monotonic() - t0 < 1.5:
                try:
                    slow_sock.sendall(b"X-Slow: 1\r\n")
                    slow_sock.settimeout(0.02)
                    try:
                        data = slow_sock.recv(1024)
                        if data == b"":
                            closed_by_server = True
                            break
                    except TimeoutError:
                        pass
                except OSError:
                    closed_by_server = True
                    break
                time.sleep(0.06)
        elapsed = time.monotonic() - t0
        self.assertTrue(closed_by_server, "Slowloris connection was not closed by header deadline")
        self.assertLess(elapsed, 1.5)

        # After slowloris release, normal requests succeed cleanly.
        status, _, _ = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 200)

    def test_invalid_backend_and_exporter_url_schemes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BackendConfig(id="typo-https", base_url="htps://127.0.0.1:8081/v1")
        with self.assertRaises(ValueError):
            BackendConfig(id="bad-ftp", base_url="ftp://127.0.0.1:8081/v1")
        with self.assertRaises(ValueError):
            BackendConfig(
                id="bad-exp",
                base_url="http://127.0.0.1:8081/v1",
                tpu_exporter_url="htps://127.0.0.1:9400/metrics",
            )
        with self.assertRaises(ValueError):
            _open_http_connection("htps", "127.0.0.1", 8081, 1.0)

        tele = scrape_tpu_exporter("htps://127.0.0.1:9400/metrics")
        self.assertFalse(tele.reachable)
        self.assertFalse(tele.exporter_up)

        bad_url_cfg = Path(self._tmpdir.name) / "bad_scheme.json"
        bad_url_cfg.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "htps://127.0.0.1:8081/v1",
                            "api_key": "never-send-in-cleartext",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_url_cfg)])
        self.assertNotEqual(rc, 0)

    def test_premature_upstream_stream_eof_signals_error_and_records_502(self) -> None:
        spot = self._track(_MockOpenAIBackend("spot-trunc", ["m-trunc"]))
        backup = self._track(_MockOpenAIBackend("backup-no-replay", ["m-trunc"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-trunc",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=spot.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="backup-no-replay",
                    type="cloud-tpu-v6e",
                    spot=False,
                    base_url=backup.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Spot sends one token without finish_reason or [DONE], then drops socket.
        partial_chunk = json.dumps(
            {
                "id": "chatcmpl-trunc",
                "object": "chat.completion.chunk",
                "model": "m-trunc",
                "choices": [
                    {"index": 0, "delta": {"content": "Partial "}, "finish_reason": None}
                ],
            }
        )
        spot.raw_sse_bytes = f"data: {partial_chunk}\n\n".encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=5.0)
        try:
            body_bytes = json.dumps(
                {
                    "model": "m-trunc",
                    "stream": True,
                    "messages": [{"role": "user", "content": "truncate mid-stream"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body_bytes,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.getheader("Transfer-Encoding"), "chunked")
            with self.assertRaises(http.client.IncompleteRead) as ctx:
                resp.read()
            partial_text = ctx.exception.partial.decode("utf-8", errors="replace")
            self.assertIn("Partial ", partial_text)
            self.assertIn("upstream_stream_interrupted", partial_text)
        finally:
            conn.close()

        # Must NOT replay a committed stream on the backup backend.
        self.assertEqual(len(backup.requests_seen), 0)

        # Router metrics must record 502 error status, not 200 success.
        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="m-trunc",backend_id="spot-trunc",status="502"} 1',
            metrics_txt,
        )
        self.assertNotIn(
            'freetpu_router_requests_total{model="m-trunc",backend_id="spot-trunc",status="200"}',
            metrics_txt,
        )

    def test_failover_reconsiders_worker_freed_during_primary_attempt(self) -> None:
        primary = self._track(
            _MockOpenAIBackend("slow-primary-fail", ["m-reconsider"], status_code=200)
        )
        fallback = self._track(
            _MockOpenAIBackend("freed-fallback", ["m-reconsider"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="slow-primary-fail",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="freed-fallback",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=fallback.base_url,
                    max_inflight=1,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        primary.status_code = 503
        primary.latency_s = 0.20

        # Occupy fallback's only slot BEFORE the request arrives (so initial selection skips it).
        fallback_state = router.engine._backends[1]
        self.assertIsNotNone(router.engine.try_reserve_backend(fallback_state))

        def release_later() -> None:
            time.sleep(0.06)
            router.engine.release_backend(fallback_state)

        releaser = threading.Thread(target=release_later, daemon=True)
        releaser.start()

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-reconsider", "messages": [{"role": "user", "content": "retry"}]},
        )
        releaser.join(timeout=1.0)

        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "freed-fallback")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:freed-fallback", json.loads(body)["choices"][0]["message"]["content"])

    def test_sse_keepalive_storm_hits_pre_output_deadline_and_fails_over(self) -> None:
        stalled_spot = self._track(
            _MockOpenAIBackend("stalled-keepalive-spot", ["m-storm"], status_code=200)
        )
        healthy = self._track(
            _MockOpenAIBackend("healthy-storm-backup", ["m-storm"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="stalled-keepalive-spot",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=stalled_spot.base_url,
                    timeout_s=0.30,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="healthy-storm-backup",
                    type="cloud-tpu-v6e",
                    spot=False,
                    base_url=healthy.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Emit keepalives every 0.05s (< 0.30s socket timeout) so only an elapsed deadline triggers.
        stalled_spot.keepalive_storm_interval_s = 0.05

        t0 = time.monotonic()
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-storm",
                "stream": True,
                "messages": [{"role": "user", "content": "escape keepalive storm"}],
            },
        )
        elapsed = time.monotonic() - t0

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-storm-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("healthy-storm-backup", body.decode("utf-8"))

    def test_tpu_exporter_zero_devices_marks_backend_unready(self) -> None:
        zero_exp = self._track(
            _MockTpuExporter(
                raw_body=(
                    "freetpu_exporter_up 1\n"
                    "freetpu_tpu_devices_discovered 0\n"
                )
            )
        )
        good_exp = self._track(_MockTpuExporter(health_status=1))
        zero_tpu = self._track(_MockOpenAIBackend("tpu-zero-dev", ["m-zero"]))
        good_tpu = self._track(_MockOpenAIBackend("tpu-has-dev", ["m-zero"]))

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            backends=[
                BackendConfig(
                    id="tpu-zero-dev",
                    type="cloud-tpu-v5e",
                    base_url=zero_tpu.base_url,
                    tpu_exporter_url=zero_exp.metrics_url,
                    cost_per_1m_prompt_usd=0.05,
                    cost_per_1m_completion_usd=0.05,
                ),
                BackendConfig(
                    id="tpu-has-dev",
                    type="cloud-tpu-v6e",
                    base_url=good_tpu.base_url,
                    tpu_exporter_url=good_exp.metrics_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        _, _, hbody = _http_request(rport, "GET", "/health")
        hjson = json.loads(hbody)
        zero_snap = next(b for b in hjson["backends"] if b["id"] == "tpu-zero-dev")
        self.assertFalse(zero_snap["healthy"])
        self.assertEqual(zero_snap["unhealthy_reason"], "tpu_no_devices_discovered")

        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-zero", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "tpu-has-dev")

    def test_tokenized_prompt_arrays_counted_in_fallback_usage(self) -> None:
        worker = self._track(
            _MockOpenAIBackend(
                "tok-worker",
                ["m-tok"],
                custom_sse_chunks=[
                    {
                        "id": "cmpl-tok",
                        "object": "text_completion",
                        "model": "m-tok",
                        "choices": [{"index": 0, "text": "done", "finish_reason": "stop"}],
                    }
                ],
            )
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "tok_usage.jsonl", callback=captured.append
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="tok-worker",
                    type="cloud-tpu-v5e",
                    base_url=worker.base_url,
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=1.0,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        # Single token-ID array (5 tokens)
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={"model": "m-tok", "stream": True, "prompt": [101, 202, 303, 404, 505]},
        )
        self.assertEqual(status, 200)

        # Batch of token-ID arrays (3 + 4 = 7 tokens)
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-tok",
                "stream": True,
                "prompt": [[1, 2, 3], [4, 5, 6, 7]],
            },
        )
        self.assertEqual(status, 200)

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0].prompt_tokens, 5)
        self.assertEqual(captured[1].prompt_tokens, 7)
        self.assertGreater(captured[0].estimated_cost_usd, 0.0)
        self.assertGreater(captured[1].estimated_cost_usd, 0.0)

    def test_multi_choice_stream_truncation_detected_when_only_choice_0_finishes(self) -> None:
        spot = self._track(_MockOpenAIBackend("spot-multi-trunc", ["m-mc"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-multi-trunc",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=spot.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # n=2 requested: choice 0 finishes, choice 1 only emits partial text, then EOF without [DONE].
        ev0 = json.dumps(
            {
                "id": "chatcmpl-mc",
                "object": "chat.completion.chunk",
                "model": "m-mc",
                "choices": [{"index": 0, "delta": {"content": "done0"}, "finish_reason": "stop"}],
            }
        )
        ev1_partial = json.dumps(
            {
                "id": "chatcmpl-mc",
                "object": "chat.completion.chunk",
                "model": "m-mc",
                "choices": [{"index": 1, "delta": {"content": "part1"}, "finish_reason": None}],
            }
        )
        spot.raw_sse_bytes = f"data: {ev0}\n\ndata: {ev1_partial}\n\n".encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=5.0)
        try:
            body_bytes = json.dumps(
                {
                    "model": "m-mc",
                    "stream": True,
                    "n": 2,
                    "messages": [{"role": "user", "content": "two choices"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body_bytes,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as ctx:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", ctx.exception.partial)
        finally:
            conn.close()

        # Worker must be placed in cooldown and metric recorded as 502, not 200.
        self.assertGreater(router.engine._backends[0].cooldown_until_monotonic, time.monotonic())
        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="m-mc",backend_id="spot-multi-trunc",status="502"} 1',
            metrics_txt,
        )

        # When both choice 0 and choice 1 finish (even without [DONE]), stream completes cleanly.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        ev1_done = json.dumps(
            {
                "id": "chatcmpl-mc",
                "object": "chat.completion.chunk",
                "model": "m-mc",
                "choices": [{"index": 1, "delta": {"content": "done1"}, "finish_reason": "stop"}],
            }
        )
        spot.raw_sse_bytes = f"data: {ev0}\n\ndata: {ev1_done}\n\n".encode("utf-8")
        status, _, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-mc",
                "stream": True,
                "n": 2,
                "messages": [{"role": "user", "content": "two complete choices"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("done1", body.decode("utf-8"))

    def test_retryable_status_fails_over_immediately_without_draining_stalled_body(self) -> None:
        release_event = threading.Event()
        stalled_primary = self._track(
            _MockOpenAIBackend(
                "stalled-503-primary",
                ["m-stall503"],
                status_code=200,
                stalled_error_release_event=release_event,
            )
        )
        healthy_backup = self._track(
            _MockOpenAIBackend("fast-backup", ["m-stall503"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="stalled-503-primary",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=stalled_primary.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="fast-backup",
                    type="cloud-tpu-v6e",
                    base_url=healthy_backup.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        stalled_primary.status_code = 503
        stalled_primary.stalled_error_body = True

        t0 = time.monotonic()
        try:
            status, hdrs, body = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": "m-stall503", "messages": [{"role": "user", "content": "fast"}]},
            )
        finally:
            release_event.set()
        elapsed = time.monotonic() - t0

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(hdrs["x-freetpu-backend"], "fast-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:fast-backup", json.loads(body)["choices"][0]["message"]["content"])

        # When sole candidate stalls mid-503-body, router synthesizes 503 JSON promptly.
        release_event.clear()
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        router.engine._backends[1].healthy = False
        t1 = time.monotonic()
        try:
            status, _, body = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": "m-stall503", "messages": [{"role": "user", "content": "sole"}]},
            )
        finally:
            release_event.set()
        elapsed2 = time.monotonic() - t1

        self.assertEqual(status, 503)
        self.assertLess(elapsed2, 1.0)
        self.assertEqual(json.loads(body)["error"]["code"], "backend_unavailable")

    def test_sse_pre_output_enforces_byte_cap_and_deadline_without_newlines(self) -> None:
        bad_primary = self._track(_MockOpenAIBackend("no-nl-primary", ["m-nonl"]))
        healthy = self._track(_MockOpenAIBackend("good-nonl-backup", ["m-nonl"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            max_sse_pre_commit_bytes=1024,
            backends=[
                BackendConfig(
                    id="no-nl-primary",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=bad_primary.base_url,
                    timeout_s=0.30,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="good-nonl-backup",
                    type="cloud-tpu-v6e",
                    base_url=healthy.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Partial line exceeding max_sse_pre_commit_bytes (1024) with ZERO newlines ->
        #    returns request-scoped 502 WITHOUT cooling down primary or failing over to paid backup.
        bad_primary.raw_sse_bytes = b":" + (b"x" * 4096)
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-nonl",
                "stream": True,
                "messages": [{"role": "user", "content": "byte cap without newline"}],
            },
        )
        self.assertEqual(status, 502)
        self.assertEqual(hdrs["x-freetpu-backend"], "no-nl-primary")
        self.assertEqual(hdrs["x-freetpu-failovers"], "0")
        self.assertIn("response_too_large", body.decode("utf-8"))
        self.assertEqual(len(healthy.requests_seen), 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
        self.assertTrue(router.engine._backends[0].healthy)

        # 2. Slow trickle without newlines (1 byte every 0.05s) hits 0.30s elapsed deadline -> fail over.
        bad_primary.raw_sse_bytes = None
        bad_primary.newline_free_trickle_interval_s = 0.05
        t0 = time.monotonic()
        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-nonl",
                "stream": True,
                "messages": [{"role": "user", "content": "deadline without newline"}],
            },
        )
        elapsed = time.monotonic() - t0
        self.assertEqual(status, 200)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(hdrs["x-freetpu-backend"], "good-nonl-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("good-nonl-backup", body.decode("utf-8"))

    def test_non_loopback_plaintext_http_with_backend_credentials_rejected(self) -> None:
        # Non-loopback HTTP with api_key or api_key_env must fail closed by default.
        with self.assertRaises(ValueError):
            BackendConfig(
                id="remote-http-key",
                base_url="http://10.0.0.5:8081/v1",
                api_key="secret-token",
            )
        with self.assertRaises(ValueError):
            BackendConfig(
                id="remote-http-env",
                base_url="http://tpu-worker.example.com:8081/v1",
                api_key_env="FREETPU_CLOUD_BACKEND_TOKEN",
            )
        # Hostname starting with "127." is NOT loopback.
        with self.assertRaises(ValueError):
            BackendConfig(
                id="spoofed-127-host",
                base_url="http://127.example.com:8081/v1",
                api_key="secret-token",
            )

        bad_cfg_path = Path(self._tmpdir.name) / "remote_http_auth.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "remote-tpu",
                            "base_url": "http://192.168.1.50:8083/v1",
                            "api_key_env": "FREETPU_CLOUD_BACKEND_TOKEN",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("allow_insecure_http", stderr_buf.getvalue())

        # Allowed cases: loopback HTTP, HTTPS, explicit allow_insecure_http=True, or unauthenticated HTTP.
        for valid_url in (
            "http://127.0.0.1:8081/v1",
            "http://127.0.0.2:8081/v1",
            "http://localhost:8081/v1",
            "http://[::1]:8081/v1",
            "https://tpu-worker.example.com/v1",
        ):
            cfg = BackendConfig(id="ok-backend", base_url=valid_url, api_key="secret")
            self.assertEqual(cfg.api_key, "secret")

        opted_in = BackendConfig(
            id="opted-in-http",
            base_url="http://10.0.0.5:8081/v1",
            api_key="secret",
            allow_insecure_http=True,
        )
        self.assertTrue(opted_in.allow_insecure_http)

        unauth_remote = BackendConfig(
            id="unauth-remote",
            base_url="http://10.0.0.5:8081/v1",
        )
        self.assertIsNone(unauth_remote.api_key)

    def test_non_boolean_backend_flags_rejected(self) -> None:
        # Quoted "false" in JSON must NOT coerce via bool("false") == True.
        with self.assertRaises(ValueError):
            BackendConfig.from_dict(
                {
                    "id": "quoted-optout",
                    "base_url": "http://10.0.0.5:8081/v1",
                    "api_key": "secret",
                    "allow_insecure_http": "false",
                }
            )
        for flag_name, bad_val in (
            ("allow_insecure_http", "true"),
            ("allow_insecure_http", 1),
            ("enabled", "false"),
            ("spot", "false"),
            ("local", "true"),
            ("discover_models", "false"),
        ):
            with self.assertRaises(ValueError):
                BackendConfig.from_dict(
                    {
                        "id": f"bad-{flag_name}",
                        "base_url": "http://127.0.0.1:8081/v1",
                        flag_name: bad_val,
                    }
                )

        # Direct Python instantiation with non-bool must also fail closed.
        with self.assertRaises(ValueError):
            BackendConfig(
                id="direct-bad-bool",
                base_url="http://10.0.0.5:8081/v1",
                api_key="secret",
                allow_insecure_http="false",  # type: ignore[arg-type]
            )

        bad_cfg_path = Path(self._tmpdir.name) / "quoted_insecure_http.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "remote-tpu",
                            "base_url": "http://10.0.0.5:8081/v1",
                            "api_key": "secret",
                            "allow_insecure_http": "false",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("JSON boolean", stderr_buf.getvalue())

    def test_pre_commit_sse_error_event_classification_and_failover(self) -> None:
        primary = self._track(_MockOpenAIBackend("vllm-primary", ["m-sse-err"]))
        backup = self._track(_MockOpenAIBackend("vllm-backup", ["m-sse-err"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="cheapest",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="vllm-primary",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=primary.base_url,
                    cost_per_1m_prompt_usd=0.10,
                    cost_per_1m_completion_usd=0.10,
                ),
                BackendConfig(
                    id="vllm-backup",
                    type="cloud-tpu-v6e",
                    base_url=backup.base_url,
                    cost_per_1m_prompt_usd=0.50,
                    cost_per_1m_completion_usd=0.50,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Retryable 500 error inside HTTP 200 SSE stream -> fails over to healthy backup.
        server_err = json.dumps(
            {
                "error": {
                    "message": "vLLM worker internal generation error",
                    "type": "server_error",
                    "code": 500,
                }
            }
        )
        primary.raw_sse_bytes = f"data: {server_err}\n\ndata: [DONE]\n\n".encode("utf-8")

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse-err",
                "stream": True,
                "messages": [{"role": "user", "content": "failover on 500 sse"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "vllm-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("vllm-backup", body.decode("utf-8"))

        # 2. Non-retryable 400 client error inside HTTP 200 SSE stream -> passed through as 400
        #    without failing over or cooling down primary.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        backup_before = len(backup.requests_seen)
        client_err = json.dumps(
            {
                "error": {
                    "message": "maximum context length exceeded",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            }
        )
        primary.raw_sse_bytes = f"data: {client_err}\n\ndata: [DONE]\n\n".encode("utf-8")

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse-err",
                "stream": True,
                "messages": [{"role": "user", "content": "too long"}],
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(hdrs["x-freetpu-backend"], "vllm-primary")
        self.assertEqual(hdrs["x-freetpu-failovers"], "0")
        self.assertIn("context length exceeded", body.decode("utf-8"))
        self.assertEqual(len(backup.requests_seen), backup_before)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

    def test_committed_stream_expires_on_post_token_keepalive_stall(self) -> None:
        stalling = self._track(
            _MockOpenAIBackend(
                "stalling-after-tok1",
                ["m-poststall"],
                post_token_keepalive_interval_s=0.05,
            )
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="stalling-after-tok1",
                    type="cloud-tpu-v5e",
                    base_url=stalling.base_url,
                    timeout_s=0.30,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        t0 = time.monotonic()
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body_bytes = json.dumps(
                {
                    "model": "m-poststall",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stall after tok1"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body_bytes,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as ctx:
                resp.read()
            self.assertIn(b"Hello ", ctx.exception.partial)
            self.assertIn(b"upstream_stream_interrupted", ctx.exception.partial)
        finally:
            conn.close()
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 1.5)
        time.sleep(0.05)
        self.assertEqual(router.engine._backends[0].inflight, 0)
        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        self.assertIn(
            'freetpu_router_requests_total{model="m-poststall",backend_id="stalling-after-tok1",status="502"} 1',
            metrics_raw.decode("utf-8"),
        )

    def test_streamed_timings_per_token_records_latest_cumulative_count(self) -> None:
        timings_chunks = [
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [{"index": 0, "delta": {"content": "one "}, "finish_reason": None}],
                "timings": {"prompt_n": 30, "predicted_n": 1},
            },
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [{"index": 0, "delta": {"content": "two "}, "finish_reason": None}],
                "timings": {"prompt_n": 30, "predicted_n": 2},
            },
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [{"index": 0, "delta": {"content": "three"}, "finish_reason": "stop"}],
                "timings": {"prompt_n": 30, "predicted_n": 12},
            },
        ]
        worker = self._track(
            _MockOpenAIBackend("llama-timings", ["m-timings"], custom_sse_chunks=timings_chunks)
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "timings.jsonl", callback=captured.append
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="llama-timings",
                    type="cloud-tpu-v5e",
                    base_url=worker.base_url,
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=2.0,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-timings",
                "stream": True,
                "messages": [{"role": "user", "content": "timings per token"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].prompt_tokens, 30)
        self.assertEqual(captured[0].completion_tokens, 12)
        # Cost: 30 * 1.0 / 1e6 + 12 * 2.0 / 1e6 = 0.000054
        self.assertAlmostEqual(captured[0].estimated_cost_usd, 0.000054, places=8)

        # Explicit OpenAI usage chunk overrides earlier timings and is NOT overwritten
        # by later timings frames arriving after usage.
        worker.custom_sse_chunks = timings_chunks + [
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15},
            },
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [],
                "timings": {"prompt_n": 99, "predicted_n": 99},
            },
        ]
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-timings",
                "stream": True,
                "messages": [{"role": "user", "content": "usage beats timings"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[1].prompt_tokens, 30)
        self.assertEqual(captured[1].completion_tokens, 15)
        # Cost: 30 * 1.0 / 1e6 + 15 * 2.0 / 1e6 = 0.000060
        self.assertAlmostEqual(captured[1].estimated_cost_usd, 0.000060, places=8)

    def test_long_prompt_progress_stream_beyond_64_events_succeeds(self) -> None:
        primary = self._track(_MockOpenAIBackend("llama-progress", ["m-prog"]))
        backup = self._track(_MockOpenAIBackend("paid-backup", ["m-prog"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="llama-progress",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="paid-backup",
                    type="cloud-tpu-v6e",
                    local=False,
                    base_url=backup.base_url,
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=2.0,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Emit 80 prompt_progress events (> old 64-event cap) before the first token.
        parts = [
            f'data: {json.dumps({"prompt_progress": {"total": 80, "processed": i}, "choices": []})}\n\n'
            for i in range(1, 81)
        ]
        final_chunk = json.dumps(
            {
                "id": "chatcmpl-prog",
                "object": "chat.completion.chunk",
                "model": "m-prog",
                "choices": [{"index": 0, "delta": {"content": "done-after-80"}, "finish_reason": "stop"}],
            }
        )
        parts.append(f"data: {final_chunk}\n\n")
        parts.append("data: [DONE]\n\n")
        primary.raw_sse_bytes = "".join(parts).encode("utf-8")

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-prog",
                "stream": True,
                "messages": [{"role": "user", "content": "long prompt"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "llama-progress")
        self.assertEqual(hdrs["x-freetpu-failovers"], "0")
        self.assertIn("done-after-80", body.decode("utf-8"))
        self.assertEqual(len(backup.requests_seen), 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

    def test_unterminated_final_sse_event_discarded_and_flagged_truncated(self) -> None:
        spot = self._track(_MockOpenAIBackend("spot-no-blank-eof", ["m-unterm"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="spot-no-blank-eof",
                    type="cloud-tpu-v5e",
                    spot=True,
                    base_url=spot.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        ev1 = json.dumps(
            {
                "id": "chatcmpl-unterm",
                "object": "chat.completion.chunk",
                "model": "m-unterm",
                "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}],
            }
        )
        ev2_unterminated = json.dumps(
            {
                "id": "chatcmpl-unterm",
                "object": "chat.completion.chunk",
                "model": "m-unterm",
                "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}],
            }
        )
        # Note ONLY ONE trailing '\n' on ev2 (missing SSE terminating blank line '\n\n')!
        spot.raw_sse_bytes = f"data: {ev1}\n\ndata: {ev2_unterminated}\n".encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=5.0)
        try:
            body_bytes = json.dumps(
                {
                    "model": "m-unterm",
                    "stream": True,
                    "messages": [{"role": "user", "content": "missing blank line at eof"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body_bytes,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as ctx:
                resp.read()
            self.assertIn(b"Hello ", ctx.exception.partial)
            self.assertIn(b"upstream_stream_interrupted", ctx.exception.partial)
        finally:
            conn.close()

        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        metrics_txt = metrics_raw.decode("utf-8")
        self.assertIn(
            'freetpu_router_requests_total{model="m-unterm",backend_id="spot-no-blank-eof",status="502"} 1',
            metrics_txt,
        )
        self.assertNotIn(
            'freetpu_router_requests_total{model="m-unterm",backend_id="spot-no-blank-eof",status="200"}',
            metrics_txt,
        )

    def test_probe_and_exporter_enforce_elapsed_deadline_on_slow_drip(self) -> None:
        slow_backend = self._track(
            _MockOpenAIBackend(
                "slow-drip-probe",
                ["m-probe"],
                slow_drip_probe_interval_s=0.05,
            )
        )
        healthy_backend = self._track(_MockOpenAIBackend("fast-probe", ["m-probe"]))

        # Also test a slow-drip TPU exporter that trickles 1 byte every 0.05s.
        class _SlowDripExporterHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Connection", "close")
                self.end_headers()
                deadline = time.monotonic() + 3.0
                try:
                    while time.monotonic() < deadline:
                        self.wfile.write(b"#")
                        self.wfile.flush()
                        time.sleep(0.05)
                except OSError:
                    pass

        slow_exp = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _SlowDripExporterHandler))
        slow_exp_url = f"http://127.0.0.1:{slow_exp.server_address[1]}/metrics"

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            probe_timeout_s=0.25,
            backends=[
                BackendConfig(
                    id="slow-drip-probe",
                    type="pixel-tpu",
                    local=True,
                    base_url=slow_backend.base_url,
                ),
                BackendConfig(
                    id="slow-exporter-worker",
                    type="cloud-tpu-v5e",
                    base_url=healthy_backend.base_url,
                    tpu_exporter_url=slow_exp_url,
                ),
                BackendConfig(
                    id="fast-probe",
                    type="cloud-tpu-v6e",
                    base_url=healthy_backend.base_url,
                ),
            ],
        )

        t0 = time.monotonic()
        router = self._track(create_router_server(cfg, probe_on_start=True))
        elapsed = time.monotonic() - t0
        rport = router.server_address[1]

        # Must finish initial probe promptly instead of hanging for 3s per slow worker.
        self.assertLess(elapsed, 1.8)
        status, _, body = _http_request(rport, "GET", "/health")
        self.assertEqual(status, 200)
        hjson = json.loads(body)
        snaps = {b["id"]: b for b in hjson["backends"]}
        self.assertFalse(snaps["slow-drip-probe"]["healthy"])
        self.assertFalse(snaps["slow-exporter-worker"]["healthy"])
        self.assertTrue(snaps["fast-probe"]["healthy"])

    def test_cli_rejects_raw_bearer_token_and_accepts_bearer_token_env(self) -> None:
        parser = _build_cli_parser()
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                parser.parse_args(["--config", "c.json", "--bearer-token", "leaked-in-argv"])
        self.assertNotEqual(ctx.exception.code, 0)

        cfg_path = Path(self._tmpdir.name) / "cli_env_auth.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        import os

        env_key = "FREETPU_TEST_CLI_BEARER_TOKEN"
        os.environ[env_key] = "env-secret-value"
        try:
            rc = router_main(
                ["--check-config", "--config", str(cfg_path), "--bearer-token-env", env_key]
            )
            self.assertEqual(rc, 0)
            cfg = load_config(cfg_path)
            cfg.auth = {"type": "bearer", "tokens_env": [env_key]}
            auth_hook, _, _ = build_hooks_from_config(cfg)
            self.assertTrue(
                auth_hook.authenticate(
                    {"Authorization": "Bearer env-secret-value"}, "/v1/models", "GET"
                ).authorized
            )
            self.assertFalse(
                auth_hook.authenticate({}, "/v1/models", "GET").authorized
            )
        finally:
            os.environ.pop(env_key, None)

    def test_cors_preflight_allows_openai_js_sdk_headers(self) -> None:
        backend = self._track(_MockOpenAIBackend("cors-sdk", ["m-sdk"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            cors_allowed_origins=["https://sdk-app.example"],
            backends=[
                BackendConfig(
                    id="cors-sdk",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, hdrs, _ = _http_request(
            rport,
            "OPTIONS",
            "/v1/chat/completions",
            headers={
                "Origin": "https://sdk-app.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type,x-stainless-retry-count",
            },
        )
        self.assertEqual(status, 204)
        allowed_hdrs = {
            h.strip().lower()
            for h in hdrs.get("access-control-allow-headers", "").split(",")
            if h.strip()
        }
        for required in (
            "authorization",
            "content-type",
            "x-freetpu-policy",
            "x-stainless-retry-count",
            "x-stainless-os",
            "x-stainless-lang",
            "openai-organization",
            "openai-project",
        ):
            self.assertIn(required, allowed_hdrs)

    def test_stream_finalizes_immediately_after_done_when_upstream_stays_open(self) -> None:
        hold_after_done = threading.Event()
        backend = self._track(
            _MockOpenAIBackend(
                "open-after-done",
                ["m-done-open"],
                hold_after_done_event=hold_after_done,
            )
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="open-after-done",
                    type="cloud-tpu-v5e",
                    base_url=backend.base_url,
                    timeout_s=0.35,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        t0 = time.monotonic()
        try:
            status, hdrs, body = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-done-open",
                    "stream": True,
                    "messages": [{"role": "user", "content": "finish on done"}],
                },
            )
        finally:
            hold_after_done.set()
        elapsed = time.monotonic() - t0

        self.assertEqual(status, 200)
        self.assertLess(elapsed, 0.8)
        self.assertEqual(hdrs["x-freetpu-backend"], "open-after-done")
        self.assertIn("data: [DONE]", body.decode("utf-8"))
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
        _, _, metrics_raw = _http_request(rport, "GET", "/metrics")
        self.assertIn(
            'freetpu_router_requests_total{model="m-done-open",backend_id="open-after-done",status="200"} 1',
            metrics_raw.decode("utf-8"),
        )

    def test_round_robin_tracks_per_model_cursor_across_wildcard_backends(self) -> None:
        wc_a = self._track(_MockOpenAIBackend("wc-a", []))
        wc_b = self._track(_MockOpenAIBackend("wc-b", []))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="round_robin",
            backends=[
                BackendConfig(
                    id="wc-a",
                    type="cloud-tpu-v5e",
                    base_url=wc_a.base_url,
                    models=["*"],
                    discover_models=False,
                    priority=10,
                ),
                BackendConfig(
                    id="wc-b",
                    type="cloud-tpu-v5e",
                    base_url=wc_b.base_url,
                    models=["*"],
                    discover_models=False,
                    priority=20,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        seen_by_model: Dict[str, List[str]] = {"wc-model-1": [], "wc-model-2": []}
        for model_name in ("wc-model-1", "wc-model-2", "wc-model-1", "wc-model-2"):
            status, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": model_name, "messages": [{"role": "user", "content": "rr"}]},
            )
            self.assertEqual(status, 200)
            seen_by_model[model_name].append(hdrs["x-freetpu-backend"])

        # Each wildcard model must rotate independently across both backends.
        self.assertEqual(seen_by_model["wc-model-1"], ["wc-a", "wc-b"])
        self.assertEqual(seen_by_model["wc-model-2"], ["wc-a", "wc-b"])

        # Verify bounded LRU eviction caps _rr_cursors at MAX_RR_CURSORS.
        for idx in range(MAX_RR_CURSORS + 50):
            router.engine.select_candidates(f"lru-model-{idx}", policy="round_robin")
        self.assertEqual(len(router.engine._rr_cursors), MAX_RR_CURSORS)

    def test_terminal_routing_failures_emit_zero_token_usage_events(self) -> None:
        primary_503 = self._track(
            _MockOpenAIBackend("primary-503", ["m-fail-usage"], status_code=200)
        )
        backup_502 = self._track(
            _MockOpenAIBackend("backup-502", ["m-fail-usage"], status_code=200)
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "terminal_failures.jsonl",
            callback=captured.append,
        )
        auth_hook = BearerTokenAuthHook({"tok-outage": "ops-caller"})
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="primary-503",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary_503.base_url,
                ),
                BackendConfig(
                    id="backup-502",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=backup_502.base_url,
                ),
            ],
        )
        router = self._track(
            create_router_server(cfg, auth_hook=auth_hook, usage_hook=usage_hook)
        )
        rport = router.server_address[1]

        # 1. No eligible backend -> emits zero-token UsageEvent with backend_id="none", status=503.
        status, hdrs1, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok-outage"},
            body={"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 503)

        # 2. All backends fail (primary 503 -> failover -> backup 502) -> emits zero-token UsageEvent.
        primary_503.status_code = 503
        backup_502.status_code = 502
        status, hdrs2, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tok-outage"},
            body={"model": "m-fail-usage", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 502)

        self.assertEqual(len(captured), 2)
        ev1, ev2 = captured
        self.assertEqual(ev1.request_id, hdrs1["x-freetpu-request-id"])
        self.assertEqual(ev1.principal, "ops-caller")
        self.assertEqual(ev1.backend_id, "none")
        self.assertEqual(ev1.status_code, 503)
        self.assertEqual(ev1.prompt_tokens, 0)
        self.assertEqual(ev1.completion_tokens, 0)
        self.assertEqual(ev1.estimated_cost_usd, 0.0)

        self.assertEqual(ev2.request_id, hdrs2["x-freetpu-request-id"])
        self.assertEqual(ev2.principal, "ops-caller")
        self.assertEqual(ev2.backend_id, "backup-502")
        self.assertEqual(ev2.status_code, 502)
        self.assertEqual(ev2.failovers, 1)
        self.assertEqual(ev2.prompt_tokens, 0)
        self.assertEqual(ev2.completion_tokens, 0)
        self.assertEqual(ev2.estimated_cost_usd, 0.0)

    def test_round_robin_cursor_keys_are_fixed_size_hashes(self) -> None:
        wc_a = self._track(_MockOpenAIBackend("wc-hash-a", ["m-a"]))
        wc_b = self._track(_MockOpenAIBackend("wc-hash-b", ["m-b"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="round_robin",
            backends=[
                BackendConfig(
                    id="wc-hash-a",
                    type="cloud-tpu-v5e",
                    base_url=wc_a.base_url,
                    models=["*"],
                    discover_models=False,
                    priority=10,
                ),
                BackendConfig(
                    id="wc-hash-b",
                    type="cloud-tpu-v5e",
                    base_url=wc_b.base_url,
                    models=["*"],
                    discover_models=False,
                    priority=20,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        long_a = "long-model-a-" + ("x" * 180)
        long_b = "long-model-b-" + ("y" * 180)
        seen: Dict[str, List[str]] = {long_a: [], long_b: []}
        for model_name in (long_a, long_b, long_a, long_b):
            status, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": model_name, "messages": [{"role": "user", "content": "rr"}]},
            )
            self.assertEqual(status, 200)
            seen[model_name].append(hdrs["x-freetpu-backend"])

        self.assertEqual(seen[long_a], ["wc-hash-a", "wc-hash-b"])
        self.assertEqual(seen[long_b], ["wc-hash-a", "wc-hash-b"])
        # Programmatic select_candidates with huge model names also stores fixed 32-byte digests.
        router.engine.select_candidates("huge-direct-" + ("z" * 32768), policy="round_robin")
        self.assertEqual(len(router.engine._rr_cursors), 3)
        for cursor_key in router.engine._rr_cursors.keys():
            self.assertIsInstance(cursor_key, bytes)
            self.assertEqual(len(cursor_key), 32)

    def test_post_commit_sse_event_size_budget_enforced(self) -> None:
        first_tok = json.dumps(
            {
                "id": "chatcmpl-postcap",
                "object": "chat.completion.chunk",
                "model": "m-postcap",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ],
            }
        )
        # Case A: valid first token, then oversized line with no newline (>1024 bytes).
        oversized_no_nl = (
            f"data: {first_tok}\n\n".encode("utf-8")
            + b"data: "
            + (b"A" * 4096)
        )
        # Case B: valid first token, then multi-line unterminated event exceeding 1024 bytes.
        oversized_multiline = (
            f"data: {first_tok}\n\n".encode("utf-8")
            + (b"data: " + (b"B" * 100) + b"\n") * 20
        )

        for label, payload_bytes in (
            ("no_newline", oversized_no_nl),
            ("multiline", oversized_multiline),
        ):
            with self.subTest(case=label):
                backend = self._track(
                    _MockOpenAIBackend(
                        f"oversize-{label}",
                        ["m-postcap"],
                        raw_sse_bytes=payload_bytes,
                    )
                )
                cfg = RouterConfig(
                    host="127.0.0.1",
                    port=0,
                    policy="local_first",
                    max_sse_pre_commit_bytes=1024,
                    backends=[
                        BackendConfig(
                            id=f"oversize-{label}",
                            type="cloud-tpu-v5e",
                            base_url=backend.base_url,
                            timeout_s=2.0,
                        )
                    ],
                )
                router = self._track(create_router_server(cfg))
                rport = router.server_address[1]

                conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
                try:
                    req_body = json.dumps(
                        {
                            "model": "m-postcap",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    ).encode("utf-8")
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        body=req_body,
                        headers={"Content-Type": "application/json"},
                    )
                    resp = conn.getresponse()
                    self.assertEqual(resp.status, 200)
                    with self.assertRaises(http.client.IncompleteRead) as exc_info:
                        resp.read()
                    partial = exc_info.exception.partial.decode("utf-8", errors="replace")
                    self.assertIn("Hello ", partial)
                    self.assertIn("response_too_large", partial)
                finally:
                    conn.close()

                # Local post-commit event size cap is request-scoped: backend stays healthy & NOT in cooldown.
                self.assertEqual(router.engine._backends[0].inflight, 0)
                self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
                self.assertTrue(router.engine._backends[0].healthy)
                m_status, _, m_body = _http_request(rport, "GET", "/metrics")
                self.assertEqual(m_status, 200)
                self.assertIn('status="502"', m_body.decode("utf-8"))

    def test_cli_bearer_token_env_preserves_exempt_paths(self) -> None:
        backend = self._track(_MockOpenAIBackend("cli-auth-b", ["m-cli-auth"]))
        cfg_path = Path(self._tmpdir.name) / "auth_no_exempt.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": 0,
                    "policy": "local_first",
                    "auth": {
                        "type": "bearer",
                        "tokens": ["old-config-token"],
                        "exempt_paths": [],
                    },
                    "backends": [
                        {
                            "id": "cli-auth-b",
                            "base_url": backend.base_url,
                            "models": ["m-cli-auth"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        env_key = "FREETPU_TEST_CLI_PRESERVE_EXEMPT"
        prev_val = os.environ.get(env_key)
        os.environ[env_key] = "env-override-secret"
        try:
            argv = [
                "--check-config",
                "--config",
                str(cfg_path),
                "--bearer-token-env",
                env_key,
            ]
            self.assertEqual(router_main(argv), 0)

            args = _build_cli_parser().parse_args(argv)
            cfg = load_config(args.config)
            _apply_cli_overrides(cfg, args)
            cfg.validate()
            self.assertEqual(cfg.auth.get("exempt_paths"), [])
            self.assertEqual(cfg.auth.get("tokens_env"), [env_key])

            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            # /health must remain authenticated because exempt_paths=[] was preserved.
            status_unauth, _, _ = _http_request(rport, "GET", "/health")
            self.assertEqual(status_unauth, 401)

            status_old, _, _ = _http_request(
                rport,
                "GET",
                "/health",
                headers={"Authorization": "Bearer old-config-token"},
            )
            self.assertEqual(status_old, 401)

            status_ok, _, _ = _http_request(
                rport,
                "GET",
                "/health",
                headers={"Authorization": "Bearer env-override-secret"},
            )
            self.assertEqual(status_ok, 200)
        finally:
            if prev_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev_val

    def test_invalid_backend_and_router_timeouts_rejected(self) -> None:
        for bad_timeout in (0, -1, -0.5, float("nan"), float("inf"), True, False, "30"):
            with self.assertRaises(ValueError):
                BackendConfig(
                    id="bad-timeout-direct",
                    base_url="http://127.0.0.1:8081/v1",
                    timeout_s=bad_timeout,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                BackendConfig.from_dict(
                    {
                        "id": "bad-timeout-dict",
                        "base_url": "http://127.0.0.1:8081/v1",
                        "timeout_s": bad_timeout,
                    }
                )

        valid_backend = BackendConfig(id="ok-b", base_url="http://127.0.0.1:8081/v1")
        for field_name, bad_val in (
            ("probe_timeout_s", 0),
            ("probe_timeout_s", -2.0),
            ("probe_timeout_s", float("nan")),
            ("probe_timeout_s", True),
            ("request_timeout_s", 0),
            ("request_timeout_s", float("inf")),
            ("health_poll_interval_s", -1.0),
            ("failure_cooldown_s", -0.1),
            ("client_read_timeout_s", 0.0),
            ("client_header_timeout_s", -1.0),
            ("backend_idle_timeout_s", 0.0),
        ):
            with self.assertRaises(ValueError):
                RouterConfig(backends=[valid_backend], **{field_name: bad_val})  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                RouterConfig.from_dict(
                    {
                        "backends": [
                            {"id": "ok-b", "base_url": "http://127.0.0.1:8081/v1"}
                        ],
                        field_name: bad_val,
                    }
                )

        bad_cfg_path = Path(self._tmpdir.name) / "bad_backend_timeout.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "neg-timeout",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "timeout_s": -5,
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("timeout_s", stderr_buf.getvalue())

    def test_malformed_usage_and_timings_counts_fallback_without_aborting(self) -> None:
        # Direct parser edge cases.
        self.assertEqual(_safe_nonneg_int(0), 0)
        self.assertEqual(_safe_nonneg_int(42), 42)
        self.assertEqual(_safe_nonneg_int(12.0), 12)
        self.assertEqual(_safe_nonneg_int(" 19 "), 19)
        for bad in (None, True, False, -1, 1.5, float("nan"), float("inf"), "N/A", "²", {}):
            self.assertIsNone(_safe_nonneg_int(bad))

        non_stream_payload = {
            "id": "chatcmpl-malformed-usage",
            "object": "chat.completion",
            "model": "m-malformed",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Completed answer with twenty characters!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": "N/A", "completion_tokens": -4},
            "timings": {"prompt_n": True, "predicted_n": "bad"},
        }
        sse_tok = json.dumps(
            {
                "id": "chatcmpl-malformed-sse",
                "object": "chat.completion.chunk",
                "model": "m-malformed",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Streamed text that must still succeed"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": {}, "completion_tokens": "oops"},
                "timings": {"prompt_n": -1, "predicted_n": False},
            }
        )
        backend = self._track(
            _MockOpenAIBackend(
                "malformed-counts-b",
                ["m-malformed"],
                custom_non_stream_payload=non_stream_payload,
                raw_sse_bytes=f"data: {sse_tok}\n\ndata: [DONE]\n\n".encode("utf-8"),
            )
        )
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "malformed_counts.jsonl",
            callback=captured.append,
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="malformed-counts-b",
                    type="cloud-tpu-v5e",
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        # 1. Non-streaming response succeeds with HTTP 200 and estimated token fallback.
        status_ns, _, body_ns = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-malformed",
                "messages": [{"role": "user", "content": "hello malformed usage"}],
            },
        )
        self.assertEqual(status_ns, 200)
        self.assertIn("Completed answer", body_ns.decode("utf-8"))

        # 2. Streaming response succeeds cleanly with HTTP 200 and estimated token fallback.
        status_s, _, body_s = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-malformed",
                "stream": True,
                "messages": [{"role": "user", "content": "hello malformed sse usage"}],
            },
        )
        self.assertEqual(status_s, 200)
        self.assertIn("Streamed text that must still succeed", body_s.decode("utf-8"))
        self.assertIn("data: [DONE]", body_s.decode("utf-8"))

        self.assertEqual(len(captured), 2)
        for ev in captured:
            self.assertEqual(ev.status_code, 200)
            self.assertGreater(ev.prompt_tokens, 0)
            self.assertGreater(ev.completion_tokens, 0)

    def test_non_streaming_response_enforces_byte_cap_and_elapsed_deadline_with_failover(
        self,
    ) -> None:
        valid_b = BackendConfig(id="ok-b", base_url="http://127.0.0.1:8081/v1")
        with self.assertRaises(ValueError):
            RouterConfig(backends=[valid_b], max_response_body_bytes=0)

        # Case 1: Oversized non-streaming response (chunked and Content-Length) returns
        # request-scoped 502 WITHOUT putting the backend in cooldown or failing over.
        for chunked in (False, True):
            with self.subTest(oversized_chunked=chunked):
                primary = self._track(
                    _MockOpenAIBackend(
                        "oversize-primary",
                        ["m-bounded-resp"],
                        oversized_non_stream_bytes=4096,
                        oversized_non_stream_chunked=chunked,
                    )
                )
                backup = self._track(
                    _MockOpenAIBackend("healthy-backup", ["m-bounded-resp"])
                )
                cfg = RouterConfig(
                    host="127.0.0.1",
                    port=0,
                    policy="local_first",
                    max_response_body_bytes=1024,
                    backends=[
                        BackendConfig(
                            id="oversize-primary",
                            type="pixel-tpu",
                            local=True,
                            base_url=primary.base_url,
                        ),
                        BackendConfig(
                            id="healthy-backup",
                            type="cloud-tpu-v5e",
                            local=False,
                            base_url=backup.base_url,
                        ),
                    ],
                )
                router = self._track(create_router_server(cfg))
                rport = router.server_address[1]
                status, hdrs, body = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-bounded-resp",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(status, 502)
                self.assertEqual(hdrs["x-freetpu-backend"], "oversize-primary")
                self.assertEqual(hdrs["x-freetpu-failovers"], "0")
                self.assertIn("response_too_large", body.decode("utf-8"))
                self.assertEqual(len(backup.requests_seen), 0)
                self.assertEqual(router.engine._backends[0].inflight, 0)
                self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
                self.assertTrue(router.engine._backends[0].healthy)

                # Subsequent normal-sized request still routes to healthy primary without failover.
                primary.oversized_non_stream_bytes = None
                status2, hdrs2, body2 = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-bounded-resp",
                        "messages": [{"role": "user", "content": "small follow-up"}],
                    },
                )
                self.assertEqual(status2, 200)
                self.assertEqual(hdrs2["x-freetpu-backend"], "oversize-primary")
                self.assertEqual(hdrs2["x-freetpu-failovers"], "0")
                self.assertIn("reply-from:oversize-primary", body2.decode("utf-8"))

        # Case 2: Slow-drip non-streaming response exceeds elapsed deadline and fails over cleanly.
        drip_primary = self._track(
            _MockOpenAIBackend(
                "drip-primary",
                ["m-drip-resp"],
                slow_drip_non_stream_interval_s=0.05,
            )
        )
        drip_backup = self._track(_MockOpenAIBackend("drip-backup", ["m-drip-resp"]))
        cfg_drip = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="drip-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=drip_primary.base_url,
                    timeout_s=0.25,
                ),
                BackendConfig(
                    id="drip-backup",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=drip_backup.base_url,
                    timeout_s=2.0,
                ),
            ],
        )
        router_drip = self._track(create_router_server(cfg_drip))
        rport_drip = router_drip.server_address[1]
        t0 = time.monotonic()
        status, hdrs, body = _http_request(
            rport_drip,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-drip-resp",
                "messages": [{"role": "user", "content": "slow drip"}],
            },
        )
        elapsed = time.monotonic() - t0
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "drip-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertLess(elapsed, 1.5)
        self.assertEqual(router_drip.engine._backends[0].inflight, 0)

    def test_explicit_health_path_resolves_independently_of_api_version(self) -> None:
        self.assertEqual(
            _join_backend_url(
                "http://127.0.0.1:8000/v1", "/health", strip_api_version=True
            )[3],
            "/health",
        )
        self.assertEqual(
            _join_backend_url(
                "http://proxy.local/vllm/v1", "/health", strip_api_version=True
            )[3],
            "/vllm/health",
        )
        self.assertEqual(
            _join_backend_url(
                "http://127.0.0.1:8000/v1", "/v1/models", strip_api_version=True
            )[3],
            "/v1/models",
        )
        self.assertEqual(
            _join_backend_url(
                "http://127.0.0.1:8000/v1",
                "/v1/chat/completions",
                strip_api_version=False,
            )[3],
            "/v1/chat/completions",
        )

        # End-to-end: mock backend serves /health (200) and /v1/models (200), while /v1/health is 404.
        backend = self._track(_MockOpenAIBackend("vllm-health-root", ["m-vllm"]))
        self.assertTrue(backend.base_url.endswith("/v1"))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="vllm-health-root",
                    type="cloud-tpu-v5e",
                    base_url=backend.base_url,
                    health_path="/health",
                    discover_models=True,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        self.assertIn("/health", backend.get_paths_seen)
        self.assertNotIn("/v1/health", backend.get_paths_seen)
        self.assertTrue(router.engine._backends[0].healthy)

        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-vllm", "messages": [{"role": "user", "content": "ping"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "vllm-health-root")

    def test_verbose_access_logs_escape_control_characters(self) -> None:
        backend = self._track(_MockOpenAIBackend("log-safe-b", ["m-log"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="log-safe-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, verbose=True))
        rport = router.server_address[1]

        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            with socket.create_connection(("127.0.0.1", rport), timeout=2.0) as sock:
                raw_req = (
                    b"GET /health?q=\x1b[31mRED\x7f\x9b HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Connection: close\r\n\r\n"
                )
                sock.sendall(raw_req)
                while sock.recv(4096):
                    pass

        logged = stderr_buf.getvalue()
        self.assertIn("\\x1b[31mRED\\x7f\\x9b", logged)
        self.assertNotIn("\x1b", logged)
        self.assertNotIn("\x7f", logged)
        self.assertNotIn("\x9b", logged)

    def test_streaming_request_errors_preserve_stream_true_in_usage_events(self) -> None:
        backend = self._track(_MockOpenAIBackend("err-stream-b", ["m-stream-err"]))
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(
            Path(self._tmpdir.name) / "stream_err_usage.jsonl",
            callback=captured.append,
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="err-stream-b",
                    type="cloud-tpu-v5e",
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        # 1. Streaming request hitting upstream HTTP 400 -> UsageEvent.stream is True.
        backend.status_code = 400
        status_1, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-stream-err",
                "stream": True,
                "messages": [{"role": "user", "content": "bad stream req"}],
            },
        )
        self.assertEqual(status_1, 400)

        # 2. Streaming request hitting pre-output SSE client error -> UsageEvent.stream is True.
        backend.status_code = 200
        client_err = json.dumps(
            {
                "error": {
                    "message": "context length exceeded",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            }
        )
        backend.raw_sse_bytes = f"data: {client_err}\n\ndata: [DONE]\n\n".encode("utf-8")
        status_2, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-stream-err",
                "stream": True,
                "messages": [{"role": "user", "content": "too long stream"}],
            },
        )
        self.assertEqual(status_2, 400)

        # 3. Non-streaming request hitting upstream HTTP 400 -> UsageEvent.stream is False.
        backend.status_code = 400
        backend.raw_sse_bytes = None
        status_3, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-stream-err",
                "stream": False,
                "messages": [{"role": "user", "content": "bad non-stream req"}],
            },
        )
        self.assertEqual(status_3, 400)

        self.assertEqual(len(captured), 3)
        self.assertEqual(captured[0].status_code, 400)
        self.assertTrue(captured[0].stream)
        self.assertEqual(captured[1].status_code, 400)
        self.assertTrue(captured[1].stream)
        self.assertEqual(captured[2].status_code, 400)
        self.assertFalse(captured[2].stream)

    def test_truncated_fixed_length_upstream_response_fails_over_to_backup(self) -> None:
        primary = self._track(
            _MockOpenAIBackend(
                "trunc-primary",
                ["m-trunc-fixed"],
                truncate_fixed_length_response=True,
            )
        )
        backup = self._track(_MockOpenAIBackend("healthy-backup", ["m-trunc-fixed"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="trunc-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="healthy-backup",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=backup.base_url,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-trunc-fixed",
                "messages": [{"role": "user", "content": "check truncation failover"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:healthy-backup", body.decode("utf-8"))
        self.assertEqual(router.engine._backends[0].inflight, 0)
        self.assertGreater(
            router.engine._backends[0].cooldown_until_monotonic,
            time.monotonic(),
        )

    def test_probe_and_exporter_enforce_deadline_while_reading_slow_drip_headers(
        self,
    ) -> None:
        srv = socket.create_server(("127.0.0.1", 0))
        srv.settimeout(3.0)
        port = srv.getsockname()[1]
        stop_evt = threading.Event()

        def serve_trickle_headers() -> None:
            while not stop_evt.is_set():
                try:
                    client, _ = srv.accept()
                except OSError:
                    break

                def handle_one(c: socket.socket) -> None:
                    try:
                        c.settimeout(1.0)
                        c.recv(4096)
                        c.sendall(b"HTTP/1.1 200 OK\r\n")
                        deadline = time.monotonic() + 3.0
                        idx = 0
                        while not stop_evt.is_set() and time.monotonic() < deadline:
                            c.sendall(f"X-Slow-Hdr-{idx}: val\r\n".encode("ascii"))
                            idx += 1
                            time.sleep(0.05)
                    except OSError:
                        pass
                    finally:
                        try:
                            c.close()
                        except OSError:
                            pass

                threading.Thread(target=handle_one, args=(client,), daemon=True).start()

        server_thread = threading.Thread(target=serve_trickle_headers, daemon=True)
        server_thread.start()
        try:
            # 1. TPU exporter scrape aborts strictly at elapsed deadline during slow headers.
            t0 = time.monotonic()
            telem = scrape_tpu_exporter(
                f"http://127.0.0.1:{port}/metrics", timeout_s=0.25
            )
            elapsed_exp = time.monotonic() - t0
            self.assertFalse(telem.reachable)
            self.assertLess(elapsed_exp, 1.0)

            # 2. Backend HTTP probe also aborts strictly at elapsed deadline during slow headers.
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                probe_timeout_s=0.25,
                backends=[
                    BackendConfig(
                        id="slow-hdr-backend",
                        base_url=f"http://127.0.0.1:{port}/v1",
                        models=["m-slow-hdr"],
                    )
                ],
            )
            t1 = time.monotonic()
            router = self._track(create_router_server(cfg, probe_on_start=True))
            elapsed_probe = time.monotonic() - t1
            self.assertFalse(router.engine._backends[0].healthy)
            self.assertLess(elapsed_probe, 1.0)
        finally:
            stop_evt.set()
            srv.close()

    def test_invalid_backend_and_exporter_url_ports_rejected(self) -> None:
        bad_urls = (
            "http://127.0.0.1:99999/v1",
            "http://127.0.0.1:0/v1",
            "http://127.0.0.1:abc/v1",
            "http://user:topsecret@127.0.0.1:99999/v1",
        )
        for bad_url in bad_urls:
            with self.assertRaises(ValueError) as ctx_b:
                BackendConfig(id="bad-port-b", base_url=bad_url)
            self.assertNotIn("topsecret", str(ctx_b.exception))

            with self.assertRaises(ValueError) as ctx_e:
                BackendConfig(
                    id="bad-port-e",
                    base_url="http://127.0.0.1:8081/v1",
                    tpu_exporter_url=bad_url,
                )
            self.assertNotIn("topsecret", str(ctx_e.exception))

        # Direct scrape_tpu_exporter call with invalid port fails closed without uncaught ValueError.
        bad_telem = scrape_tpu_exporter("http://127.0.0.1:99999/metrics", timeout_s=0.5)
        self.assertFalse(bad_telem.reachable)
        self.assertIn("port", (bad_telem.error or "").lower())

        # --check-config rejects invalid backend/exporter ports before runtime startup.
        bad_cfg_path = Path(self._tmpdir.name) / "bad_url_port.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "bad-port-cli",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "tpu_exporter_url": "http://user:topsecret@127.0.0.1:70000/metrics",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertIn("port", stderr_buf.getvalue().lower())
        self.assertNotIn("topsecret", stderr_buf.getvalue())

    def test_malformed_backend_api_key_rejected_and_redacted_from_health(self) -> None:
        secret_tok = "sk-super-secret-leak-check-999"
        # 1. Config validation rejects control chars / newlines without echoing the secret.
        for bad_key in (f"{secret_tok}\n", f"{secret_tok}\r\nX-Inject: 1", f"{secret_tok}\x00"):
            with self.assertRaises(ValueError) as ctx_direct:
                BackendConfig(
                    id="bad-key-b",
                    base_url="http://127.0.0.1:8081/v1",
                    api_key=bad_key,
                )
            self.assertNotIn(secret_tok, str(ctx_direct.exception))

            with self.assertRaises(ValueError) as ctx_dict:
                BackendConfig.from_dict(
                    {
                        "id": "bad-key-dict",
                        "base_url": "http://127.0.0.1:8081/v1",
                        "api_key": bad_key,
                    }
                )
            self.assertNotIn(secret_tok, str(ctx_dict.exception))

        # 2. --check-config rejects malformed api_key without echoing secret to stderr.
        bad_cfg_path = Path(self._tmpdir.name) / "bad_api_key.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "bad-key-cli",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "api_key": f"{secret_tok}\n",
                            "models": ["m1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_buf = io.StringIO()
        with redirect_stderr(stderr_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg_path)])
        self.assertNotEqual(rc, 0)
        self.assertNotIn(secret_tok, stderr_buf.getvalue())

        # 3. Runtime api_key_env with trailing newline fails probe safely and NEVER leaks on /health.
        backend = self._track(_MockOpenAIBackend("env-secret-b", ["m-env-sec"]))
        env_var = "FREETPU_TEST_MALFORMED_BACKEND_KEY"
        prev_env = os.environ.get(env_var)
        os.environ[env_var] = f"{secret_tok}\n"
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="env-secret-b",
                        base_url=backend.base_url,
                        api_key_env=env_var,
                        models=["m-env-sec"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg, probe_on_start=True))
            rport = router.server_address[1]
            status, _, body = _http_request(rport, "GET", "/health")
            self.assertEqual(status, 503)
            body_text = body.decode("utf-8")
            self.assertNotIn(secret_tok, body_text)
            payload = json.loads(body_text)
            self.assertEqual(
                payload["backends"][0]["unhealthy_reason"], "invalid_request_header"
            )
        finally:
            if prev_env is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = prev_env

    def test_shared_request_memory_budget_bounds_concurrent_bodies(self) -> None:
        valid_b = BackendConfig(id="ok-b", base_url="http://127.0.0.1:8081/v1")
        with self.assertRaises(ValueError):
            RouterConfig(backends=[valid_b], max_total_request_memory_bytes=0)

        slow_b = self._track(_MockOpenAIBackend("mem-slow-b", ["m-mem"], latency_s=0.35))
        # Total budget 2000 bytes: first ~450-byte body needs 1800 to parse (fits), then retains
        # ~450 bytes while waiting on upstream; concurrent second ~450-byte body needs 1800
        # (450 + 1800 > 2000) -> immediately rejected with 503 before buffering/parsing.
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_total_request_memory_bytes=2000,
            backends=[
                BackendConfig(
                    id="mem-slow-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=slow_b.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        req_payload = {
            "model": "m-mem",
            "messages": [{"role": "user", "content": "x" * 380}],
        }
        first_result: Dict[str, Any] = {}

        def send_first() -> None:
            st, _, bd = _http_request(
                rport, "POST", "/v1/chat/completions", body=req_payload
            )
            first_result["status"] = st
            first_result["body"] = bd

        t1 = threading.Thread(target=send_first, daemon=True)
        t1.start()

        # Wait until request 1 has entered the backend (meaning its body is reserved).
        deadline = time.monotonic() + 2.0
        while slow_b.current_inflight < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(slow_b.current_inflight, 1)

        st2, _, bd2 = _http_request(
            rport, "POST", "/v1/chat/completions", body=req_payload
        )
        self.assertEqual(st2, 503)
        self.assertIn("server_memory_limit_reached", bd2.decode("utf-8"))

        t1.join(timeout=3.0)
        self.assertEqual(first_result.get("status"), 200)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

        # Once request 1 releases its budget, a subsequent request succeeds cleanly.
        slow_b.latency_s = 0.0
        st3, _, _ = _http_request(
            rport, "POST", "/v1/chat/completions", body=req_payload
        )
        self.assertEqual(st3, 200)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_sse_enforces_deadline_during_slow_chunked_framing(self) -> None:
        first_chunk_json = json.dumps(
            {
                "id": "chatcmpl-slow-frame",
                "object": "chat.completion.chunk",
                "model": "m-slow-frame",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ],
            }
        )
        first_sse = f"data: {first_chunk_json}\n\n".encode("utf-8")

        def run_framing_server(send_first_token: bool) -> Tuple[socket.socket, int, threading.Event]:
            srv = socket.create_server(("127.0.0.1", 0))
            srv.settimeout(3.0)
            stop_evt = threading.Event()

            def accept_loop() -> None:
                while not stop_evt.is_set():
                    try:
                        c, _ = srv.accept()
                    except OSError:
                        break

                    def handle_conn(sock: socket.socket) -> None:
                        try:
                            sock.settimeout(1.0)
                            req = sock.recv(4096)
                            if req.startswith(b"GET /v1/models"):
                                body = json.dumps(
                                    {"object": "list", "data": [{"id": "m-slow-frame"}]}
                                ).encode("utf-8")
                                sock.sendall(
                                    b"HTTP/1.1 200 OK\r\n"
                                    b"Content-Type: application/json\r\n"
                                    + f"Content-Length: {len(body)}\r\n".encode("ascii")
                                    + b"Connection: close\r\n\r\n"
                                    + body
                                )
                                return
                            sock.sendall(
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: text/event-stream; charset=utf-8\r\n"
                                b"Transfer-Encoding: chunked\r\n"
                                b"Connection: close\r\n\r\n"
                            )
                            if send_first_token:
                                sock.sendall(
                                    f"{len(first_sse):x}\r\n".encode("ascii")
                                    + first_sse
                                    + b"\r\n"
                                )
                            # Trickle chunk-size framing bytes 1 byte every 50ms without \r\n.
                            dl = time.monotonic() + 3.0
                            while not stop_evt.is_set() and time.monotonic() < dl:
                                sock.sendall(b"0")
                                time.sleep(0.05)
                        except OSError:
                            pass
                        finally:
                            try:
                                sock.close()
                            except OSError:
                                pass

                    threading.Thread(target=handle_conn, args=(c,), daemon=True).start()

            threading.Thread(target=accept_loop, daemon=True).start()
            return srv, srv.getsockname()[1], stop_evt

        # Case 1: Post-commit slow chunk-size framing aborts strictly at progress deadline.
        srv1, port1, stop1 = run_framing_server(send_first_token=True)
        try:
            cfg1 = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="slow-frame-post",
                        base_url=f"http://127.0.0.1:{port1}/v1",
                        timeout_s=0.30,
                    )
                ],
            )
            router1 = self._track(create_router_server(cfg1))
            rport1 = router1.server_address[1]
            t0 = time.monotonic()
            conn = http.client.HTTPConnection("127.0.0.1", rport1, timeout=3.0)
            try:
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {
                            "model": "m-slow-frame",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                with self.assertRaises(http.client.IncompleteRead) as exc_info:
                    resp.read()
                elapsed1 = time.monotonic() - t0
                partial = exc_info.exception.partial.decode("utf-8", errors="replace")
                self.assertIn("Hello ", partial)
                self.assertIn("upstream_stream_interrupted", partial)
                self.assertLess(elapsed1, 1.5)
            finally:
                conn.close()
        finally:
            stop1.set()
            srv1.close()

        # Case 2: Pre-commit slow chunk-size framing aborts strictly at deadline and fails over.
        srv2, port2, stop2 = run_framing_server(send_first_token=False)
        backup = self._track(_MockOpenAIBackend("healthy-sse-backup", ["m-slow-frame"]))
        try:
            cfg2 = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                backends=[
                    BackendConfig(
                        id="slow-frame-pre",
                        local=True,
                        base_url=f"http://127.0.0.1:{port2}/v1",
                        timeout_s=0.30,
                    ),
                    BackendConfig(
                        id="healthy-sse-backup",
                        local=False,
                        base_url=backup.base_url,
                        timeout_s=2.0,
                    ),
                ],
            )
            router2 = self._track(create_router_server(cfg2))
            rport2 = router2.server_address[1]
            t1 = time.monotonic()
            status, hdrs, body = _http_request(
                rport2,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-slow-frame",
                    "stream": True,
                    "messages": [{"role": "user", "content": "failover pre-commit"}],
                },
            )
            elapsed2 = time.monotonic() - t1
            self.assertEqual(status, 200)
            self.assertEqual(hdrs["x-freetpu-backend"], "healthy-sse-backup")
            self.assertEqual(hdrs["x-freetpu-failovers"], "1")
            self.assertLess(elapsed2, 1.5)
        finally:
            stop2.set()
            srv2.close()

    def test_default_models_probe_rejects_html_and_non_model_json(self) -> None:
        html_proxy = self._track(
            _MockOpenAIBackend(
                "html-catchall",
                ["m-proxy"],
                raw_models_response=(200, "text/html", b"<html><body>OK</body></html>"),
            )
        )
        bad_json_proxy = self._track(
            _MockOpenAIBackend(
                "non-model-json",
                ["m-proxy"],
                raw_models_response=(200, "application/json", b'{"status":"ok"}'),
            )
        )
        explicit_health_backend = self._track(
            _MockOpenAIBackend(
                "explicit-health-plain",
                ["m-proxy"],
                raw_health_response=(200, "text/plain", b"ok"),
            )
        )
        healthy_backup = self._track(_MockOpenAIBackend("good-backup", ["m-proxy"]))

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="html-catchall",
                    local=True,
                    priority=1,
                    base_url=html_proxy.base_url,
                    models=["m-proxy"],
                ),
                BackendConfig(
                    id="non-model-json",
                    local=True,
                    priority=2,
                    base_url=bad_json_proxy.base_url,
                    models=["m-proxy"],
                ),
                BackendConfig(
                    id="explicit-health-plain",
                    local=False,
                    priority=20,
                    base_url=explicit_health_backend.base_url,
                    health_path="/health",
                    models=["m-proxy"],
                ),
                BackendConfig(
                    id="good-backup",
                    local=False,
                    priority=10,
                    base_url=healthy_backup.base_url,
                    models=["m-proxy"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg, probe_on_start=True))
        rport = router.server_address[1]

        states = {b.config.id: b for b in router.engine._backends}
        # Default /v1/models probe rejects HTML and non-model JSON responses.
        self.assertFalse(states["html-catchall"].healthy)
        self.assertEqual(
            states["html-catchall"].unhealthy_reason, "invalid_models_response"
        )
        self.assertFalse(states["non-model-json"].healthy)
        self.assertEqual(
            states["non-model-json"].unhealthy_reason, "invalid_models_response"
        )
        # Explicit health_path still accepts status-only 2xx plain-text health checks.
        self.assertTrue(states["explicit-health-plain"].healthy)
        self.assertTrue(states["good-backup"].healthy)

        # Inference routes around the misrouted local proxies to good-backup.
        status, hdrs, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-proxy", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "good-backup")

    def test_cumulative_header_bytes_capped_with_431_before_auth(self) -> None:
        backend = self._track(_MockOpenAIBackend("hdr-cap-b", ["m-hdr"]))
        auth_calls: List[str] = []

        class _CountingAuth(BearerTokenAuthHook):
            def authenticate(self, headers, path, method):  # type: ignore[override]
                auth_calls.append(path)
                return super().authenticate(headers, path, method)

        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_header_bytes=2048,
            backends=[
                BackendConfig(
                    id="hdr-cap-b",
                    base_url=backend.base_url,
                    models=["m-hdr"],
                )
            ],
        )
        router = self._track(
            create_router_server(cfg, auth_hook=_CountingAuth({"secret": "u"}))
        )
        rport = router.server_address[1]

        # Many small headers totaling >2048 bytes (well under CPython's 65536/line limit).
        big_headers = "".join(f"X-Pad-{i}: " + ("a" * 120) + "\r\n" for i in range(30))
        raw_req = (
            "GET /health HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{rport}\r\n"
            + big_headers
            + "Connection: close\r\n\r\n"
        ).encode("ascii")

        with socket.create_connection(("127.0.0.1", rport), timeout=2.0) as sock:
            sock.sendall(raw_req)
            resp_chunks = []
            while True:
                try:
                    data = sock.recv(4096)
                except ConnectionResetError:
                    break
                if not data:
                    break
                resp_chunks.append(data)
        raw_resp = b"".join(resp_chunks).decode("utf-8", errors="replace")
        self.assertTrue(raw_resp.startswith("HTTP/1.1 431 "))
        self.assertIn("request_header_fields_too_large", raw_resp)
        # Rejected before authentication hook ever runs.
        self.assertEqual(auth_calls, [])

        # Normal request under cap still succeeds.
        status_ok, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(status_ok, 200)

    def test_dense_json_containers_and_deep_nesting_rejected_before_json_loads(self) -> None:
        backend = self._track(_MockOpenAIBackend("json-guard-b", ["m-jguard"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_json_depth=16,
            max_json_nodes=500,
            backends=[
                BackendConfig(
                    id="json-guard-b",
                    base_url=backend.base_url,
                    models=["m-jguard"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Dense container array exceeding max_json_nodes (500) -> HTTP 400 before json.loads.
        dense_body = (
            '{"model":"m-jguard","messages":['
            + ",".join(["[]"] * 600)
            + "]}"
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=2.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=dense_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 400)
            self.assertIn("json_complexity_limit_exceeded", raw)
        finally:
            conn.close()

        # 2. Deeply nested containers exceeding max_json_depth (16) -> HTTP 400.
        deep_body = (
            '{"model":"m-jguard","x":' + ("[" * 25) + "0" + ("]" * 25) + "}"
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=2.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=deep_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 400)
            self.assertIn("json_complexity_limit_exceeded", raw)
        finally:
            conn.close()

        # 3. Legitimate request with a large string (including brackets/commas inside quotes) passes.
        status_ok, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-jguard",
                "messages": [
                    {"role": "user", "content": ("[{},]" * 500)}
                ],
            },
        )
        self.assertEqual(status_ok, 200)
        self.assertEqual(len(backend.requests_seen), 1)

    def test_client_disconnect_during_primary_timeout_skips_fallback(self) -> None:
        primary = self._track(
            _MockOpenAIBackend(
                "slow-primary",
                ["m-cancel-retry"],
                latency_s=0.45,
            )
        )
        fallback = self._track(
            _MockOpenAIBackend("paid-fallback", ["m-cancel-retry"])
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="slow-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                    timeout_s=0.25,
                ),
                BackendConfig(
                    id="paid-fallback",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=fallback.base_url,
                    timeout_s=2.0,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        req_body = json.dumps(
            {
                "model": "m-cancel-retry",
                "messages": [{"role": "user", "content": "cancel before failover"}],
            }
        ).encode("utf-8")
        sock = socket.create_connection(("127.0.0.1", rport), timeout=2.0)
        try:
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                + f"Host: 127.0.0.1:{rport}\r\n".encode("ascii")
                + b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(req_body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + req_body
            )
            # Wait until primary has received the request, then disconnect client before timeout.
            dl = time.monotonic() + 2.0
            while len(primary.requests_seen) == 0 and time.monotonic() < dl:
                time.sleep(0.01)
            self.assertEqual(len(primary.requests_seen), 1)
        finally:
            _abort_tcp_socket(sock)

        # Wait past primary's 0.25s timeout so the retry check runs.
        time.sleep(0.45)
        self.assertEqual(len(fallback.requests_seen), 0)
        self.assertEqual(router.engine._backends[0].inflight, 0)
        self.assertEqual(router.engine._backends[1].inflight, 0)

    def test_same_origin_browser_post_allowed_with_empty_cors_allowlist(self) -> None:
        backend = self._track(_MockOpenAIBackend("same-orig-b", ["m-sameorig"]))

        # 1. Direct same-origin POST (Origin matches Host) allowed with empty cors_allowed_origins.
        cfg_direct = RouterConfig(
            host="127.0.0.1",
            port=0,
            cors_allowed_origins=[],
            backends=[
                BackendConfig(
                    id="same-orig-b",
                    base_url=backend.base_url,
                    models=["m-sameorig"],
                )
            ],
        )
        router_direct = self._track(create_router_server(cfg_direct))
        rport = router_direct.server_address[1]

        st_same, hdrs_same, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": f"http://127.0.0.1:{rport}"},
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_same, 200)
        self.assertNotIn("access-control-allow-origin", hdrs_same)

        # Loopback alias localhost:<bound_port> is also a trusted listener authority.
        st_localhost, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": f"localhost:{rport}",
                "Origin": f"http://localhost:{rport}",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_localhost, 200)

        # DNS-rebound Host matching Origin on loopback listener MUST be rejected with 403!
        st_rebind, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": f"evil.example:{rport}",
                "Origin": f"http://evil.example:{rport}",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_rebind, 403)

        # Port mismatch on loopback Host matching Origin is rejected with 403.
        wrong_port = rport + 1 if rport < 65000 else rport - 1
        st_wrong_port, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": f"127.0.0.1:{wrong_port}",
                "Origin": f"http://127.0.0.1:{wrong_port}",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_wrong_port, 403)

        # Cross-origin POST is still rejected with 403.
        st_cross, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://evil.example.com"},
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_cross, 403)

        # Scheme mismatch (https Origin on plain HTTP without trust_proxy_headers) is rejected.
        st_scheme, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={
                "Origin": f"https://127.0.0.1:{rport}",
                "X-Forwarded-Proto": "https",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_scheme, 403)

        # 2. Reverse proxy TLS termination with configured public_origin allows same-origin POST.
        cfg_pub = RouterConfig(
            host="127.0.0.1",
            port=0,
            cors_allowed_origins=[],
            public_origin="https://chat.example.com",
            backends=[
                BackendConfig(
                    id="same-orig-b",
                    base_url=backend.base_url,
                    models=["m-sameorig"],
                )
            ],
        )
        router_pub = self._track(create_router_server(cfg_pub))
        rport_pub = router_pub.server_address[1]

        st_pub_ok, _, _ = _http_request(
            rport_pub,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://chat.example.com"},
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_pub_ok, 200)

        st_pub_bad, _, _ = _http_request(
            rport_pub,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://other.example.com"},
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_pub_bad, 403)

        # 3. Even with trust_proxy_headers=True, an untrusted external Host without public_origin is rejected;
        #    trusted listener host with X-Forwarded-Proto=https succeeds.
        cfg_proxy = RouterConfig(
            host="127.0.0.1",
            port=0,
            cors_allowed_origins=[],
            trust_proxy_headers=True,
            backends=[
                BackendConfig(
                    id="same-orig-b",
                    base_url=backend.base_url,
                    models=["m-sameorig"],
                )
            ],
        )
        router_proxy = self._track(create_router_server(cfg_proxy))
        rport_proxy = router_proxy.server_address[1]

        st_proxy_rebind, _, _ = _http_request(
            rport_proxy,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": "ui.example.com",
                "Origin": "https://ui.example.com",
                "X-Forwarded-Proto": "https",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_proxy_rebind, 403)

        st_proxy_trusted, _, _ = _http_request(
            rport_proxy,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": f"127.0.0.1:{rport_proxy}",
                "Origin": f"https://127.0.0.1:{rport_proxy}",
                "X-Forwarded-Proto": "https",
            },
            body={"model": "m-sameorig", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_proxy_trusted, 200)

    def test_json_structure_scanner_escape_heavy_string_bounded_memory(self) -> None:
        # 50,000 escaped quote/backslash groups (~300 KiB): O(1) working memory, zero regex stack.
        escape_heavy = (
            '{"model":"m-esc","messages":[{"role":"user","content":"'
            + ('\\"\\\\\\"' * 50_000)
            + '"}]}'
        )
        tracemalloc.start()
        try:
            _validate_json_structure(escape_heavy, max_depth=64, max_nodes=32768)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak_bytes, 64 * 1024)

        # Live end-to-end request with escape-heavy content succeeds cleanly.
        backend = self._track(_MockOpenAIBackend("esc-b", ["m-esc"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="esc-b",
                    base_url=backend.base_url,
                    models=["m-esc"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        status, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-esc",
                "messages": [{"role": "user", "content": ('"\\"' * 2000)}],
            },
        )
        self.assertEqual(status, 200)

    def test_committed_sse_huge_usage_string_no_failover_or_duplicate_response(self) -> None:
        # 5000-digit decimal string exceeds Python's int_max_str_digits (4300) and must return None safely.
        self.assertIsNone(_safe_nonneg_int("9" * 5000))

        tok1 = json.dumps(
            {
                "id": "chatcmpl-huge-usage",
                "object": "chat.completion.chunk",
                "model": "m-huge-usage",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ],
            }
        )
        tok2 = json.dumps(
            {
                "id": "chatcmpl-huge-usage",
                "object": "chat.completion.chunk",
                "model": "m-huge-usage",
                "choices": [
                    {"index": 0, "delta": {"content": "world"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": "9" * 5000,
                    "completion_tokens": "8" * 5000,
                },
            }
        )
        primary = self._track(
            _MockOpenAIBackend(
                "primary-huge-usage",
                ["m-huge-usage"],
                raw_sse_bytes=f"data: {tok1}\n\ndata: {tok2}\n\ndata: [DONE]\n\n".encode(
                    "utf-8"
                ),
            )
        )
        backup = self._track(_MockOpenAIBackend("backup-huge-usage", ["m-huge-usage"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="primary-huge-usage",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="backup-huge-usage",
                    local=False,
                    base_url=backup.base_url,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-huge-usage",
                "stream": True,
                "messages": [{"role": "user", "content": "huge usage digits"}],
            },
        )
        text = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "primary-huge-usage")
        self.assertEqual(hdrs["x-freetpu-failovers"], "0")
        self.assertIn("Hello ", text)
        self.assertIn("world", text)
        self.assertIn("data: [DONE]", text)
        self.assertNotIn("HTTP/1.1 ", text)
        self.assertEqual(len(backup.requests_seen), 0)

    def test_close_delimited_truncated_json_200_fails_over_to_backup(self) -> None:
        primary = self._track(
            _MockOpenAIBackend(
                "trunc-json-primary",
                ["m-trunc-json"],
                close_delimited_truncated_json=True,
            )
        )
        backup = self._track(_MockOpenAIBackend("healthy-json-backup", ["m-trunc-json"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="trunc-json-primary",
                    type="pixel-tpu",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="healthy-json-backup",
                    type="cloud-tpu-v5e",
                    local=False,
                    base_url=backup.base_url,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        status, hdrs, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-trunc-json",
                "messages": [{"role": "user", "content": "truncated close-delimited json"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "healthy-json-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("reply-from:healthy-json-backup", body.decode("utf-8"))
        self.assertEqual(router.engine._backends[0].inflight, 0)
        self.assertGreater(
            router.engine._backends[0].cooldown_until_monotonic,
            time.monotonic(),
        )

    def test_shared_response_memory_budget_bounds_buffered_responses(self) -> None:
        valid_b = BackendConfig(id="ok-b", base_url="http://127.0.0.1:8081/v1")
        with self.assertRaises(ValueError):
            RouterConfig(backends=[valid_b], max_total_response_memory_bytes=0)

        # Case 1: Both Content-Length and chunked responses exceeding the shared response
        # memory budget return HTTP 503 WITHOUT putting the backend in cooldown.
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                backend = self._track(
                    _MockOpenAIBackend(
                        "resp-mem-b",
                        ["m-respmem"],
                        oversized_non_stream_bytes=1500,
                        oversized_non_stream_chunked=chunked,
                    )
                )
                cfg = RouterConfig(
                    host="127.0.0.1",
                    port=0,
                    max_response_body_bytes=65536,
                    max_total_response_memory_bytes=1200,
                    backends=[
                        BackendConfig(
                            id="resp-mem-b",
                            type="pixel-tpu",
                            local=True,
                            base_url=backend.base_url,
                        )
                    ],
                )
                router = self._track(create_router_server(cfg))
                rport = router.server_address[1]

                status, hdrs, body = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-respmem",
                        "messages": [{"role": "user", "content": "big reply"}],
                    },
                )
                self.assertEqual(status, 503)
                self.assertIn("server_memory_limit_reached", body.decode("utf-8"))
                self.assertEqual(hdrs["x-freetpu-failovers"], "0")
                self.assertEqual(router.engine._resp_mem_used_bytes, 0)
                self.assertEqual(router.engine._backends[0].inflight, 0)
                self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
                self.assertTrue(router.engine._backends[0].healthy)

                # Small response within budget succeeds cleanly on the same healthy backend.
                backend.oversized_non_stream_bytes = None
                st_ok, _, _ = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-respmem",
                        "messages": [{"role": "user", "content": "small reply"}],
                    },
                )
                self.assertEqual(st_ok, 200)
                self.assertEqual(router.engine._resp_mem_used_bytes, 0)

    def test_local_get_endpoints_reject_untrusted_host_without_origin(self) -> None:
        backend = self._track(_MockOpenAIBackend("host-guard-b", ["m-hostguard"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            auth={"type": "bearer", "tokens": ["tok-1"]},
            backends=[
                BackendConfig(
                    id="host-guard-b",
                    base_url=backend.base_url,
                    models=["m-hostguard"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # DNS-rebound Host on GET /health, /metrics, /v1/models with NO Origin header -> 403!
        for path in ("/health", "/metrics", "/v1/models"):
            st_bad, _, bd_bad = _http_request(
                rport,
                "GET",
                path,
                headers={"Host": f"evil.example:{rport}"},
            )
            self.assertEqual(st_bad, 403)
            self.assertIn("invalid_host", bd_bad.decode("utf-8"))
            self.assertNotIn(backend.base_url, bd_bad.decode("utf-8"))

        # Wrong port on loopback Host -> 403.
        wrong_port = rport + 1 if rport < 65000 else rport - 1
        st_wp, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Host": f"127.0.0.1:{wrong_port}"},
        )
        self.assertEqual(st_wp, 403)

        # Legitimate loopback Host headers succeed.
        for good_host in (f"127.0.0.1:{rport}", f"localhost:{rport}"):
            st_ok, _, _ = _http_request(
                rport,
                "GET",
                "/health",
                headers={"Host": good_host},
            )
            self.assertEqual(st_ok, 200)

        # Configured public_origin host also permitted on GET /health without Origin.
        cfg_pub = RouterConfig(
            host="127.0.0.1",
            port=0,
            public_origin="https://api.example.com",
            backends=[
                BackendConfig(
                    id="host-guard-b",
                    base_url=backend.base_url,
                    models=["m-hostguard"],
                )
            ],
        )
        router_pub = self._track(create_router_server(cfg_pub))
        rport_pub = router_pub.server_address[1]
        st_pub_ok, _, _ = _http_request(
            rport_pub,
            "GET",
            "/health",
            headers={"Host": "api.example.com"},
        )
        self.assertEqual(st_pub_ok, 200)
        st_pub_bad, _, _ = _http_request(
            rport_pub,
            "GET",
            "/health",
            headers={"Host": "other.example.com"},
        )
        self.assertEqual(st_pub_bad, 403)

    def test_oversized_and_non_string_model_ids_rejected_before_metering(self) -> None:
        backend = self._track(_MockOpenAIBackend("model-guard-b", ["m-ok"]))
        jsonl_path = Path(self._tmpdir.name) / "bounded_model_usage.jsonl"
        captured: List[UsageEvent] = []
        usage_hook = JsonlUsageMeterHook(jsonl_path, callback=captured.append)
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="model-guard-b",
                    base_url=backend.base_url,
                    models=["m-ok"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        for bad_model in ("x" * 10000, 12345, ["m-ok"], "bad\nmodel", {"id": "m"}):
            st, _, bd = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": bad_model,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st, 400)
            self.assertIn("invalid_model_id", bd.decode("utf-8"))

        # Rejected at admission before routing or writing any JSONL usage records.
        self.assertEqual(len(backend.requests_seen), 0)
        self.assertEqual(len(captured), 0)
        self.assertFalse(jsonl_path.exists())

        # Belt-and-suspenders: direct programmatic terminal failure clamps model length.
        router.engine._record_terminal_failure(
            request_id="req-clamp",
            model="z" * 1000,
            backend_id="none",
            matched_backend=False,
            status_code=503,
            latency_ms=1.0,
            stream=False,
            failovers=0,
            principal="anon",
        )
        self.assertEqual(len(captured), 1)
        self.assertLessEqual(len(captured[0].model), 256)

    def test_fixed_length_sse_premature_eof_after_finish_reason_marks_interrupted(
        self,
    ) -> None:
        finished_chunk = json.dumps(
            {
                "id": "chatcmpl-fixed-sse",
                "object": "chat.completion.chunk",
                "model": "m-fixed-sse",
                "choices": [
                    {"index": 0, "delta": {"content": "Done!"}, "finish_reason": "stop"}
                ],
            }
        )
        sent_sse = f"data: {finished_chunk}\n\n".encode("utf-8")
        declared_len = len(sent_sse) + 250  # Missing trailing usage chunk + [DONE]

        srv = socket.create_server(("127.0.0.1", 0))
        srv.settimeout(3.0)
        port = srv.getsockname()[1]
        stop_evt = threading.Event()

        def accept_loop() -> None:
            while not stop_evt.is_set():
                try:
                    c, _ = srv.accept()
                except OSError:
                    break

                def handle_one(sock: socket.socket) -> None:
                    try:
                        sock.settimeout(1.0)
                        req = sock.recv(4096)
                        if req.startswith(b"GET /v1/models"):
                            body = json.dumps(
                                {"object": "list", "data": [{"id": "m-fixed-sse"}]}
                            ).encode("utf-8")
                            sock.sendall(
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: application/json\r\n"
                                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                                + b"Connection: close\r\n\r\n"
                                + body
                            )
                            return
                        sock.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: text/event-stream; charset=utf-8\r\n"
                            + f"Content-Length: {declared_len}\r\n".encode("ascii")
                            + b"Connection: close\r\n\r\n"
                            + sent_sse
                        )
                    except OSError:
                        pass
                    finally:
                        try:
                            sock.close()
                        except OSError:
                            pass

                threading.Thread(target=handle_one, args=(c,), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="fixed-sse-trunc",
                        base_url=f"http://127.0.0.1:{port}/v1",
                        timeout_s=2.0,
                    )
                ],
            )
            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
            try:
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {
                            "model": "m-fixed-sse",
                            "stream": True,
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                with self.assertRaises(http.client.IncompleteRead) as exc_info:
                    resp.read()
                partial = exc_info.exception.partial.decode("utf-8", errors="replace")
                self.assertIn("Done!", partial)
                self.assertIn("upstream_stream_interrupted", partial)
            finally:
                conn.close()

            self.assertEqual(router.engine._backends[0].inflight, 0)
            self.assertGreater(
                router.engine._backends[0].cooldown_until_monotonic,
                time.monotonic(),
            )
            m_st, _, m_bd = _http_request(rport, "GET", "/metrics")
            self.assertEqual(m_st, 200)
            self.assertIn('status="502"', m_bd.decode("utf-8"))
        finally:
            stop_evt.set()
            srv.close()

    def test_non_boolean_stream_flag_rejected_before_routing(self) -> None:
        backend = self._track(_MockOpenAIBackend("stream-flag-b", ["m-sflag"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="stream-flag-b",
                    base_url=backend.base_url,
                    models=["m-sflag"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        for bad_stream in ("false", "true", 1, 0, [], {"enabled": True}):
            st, _, bd = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-sflag",
                    "stream": bad_stream,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st, 400)
            self.assertIn("invalid_stream_flag", bd.decode("utf-8"))

        self.assertEqual(len(backend.requests_seen), 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

        for good_stream in (None, False, True):
            st_ok, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-sflag",
                    "stream": good_stream,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st_ok, 200)

    def test_client_disconnect_mid_request_cancels_active_upstream_without_cooldown(
        self,
    ) -> None:
        srv = socket.create_server(("127.0.0.1", 0))
        srv.settimeout(3.0)
        port = srv.getsockname()[1]
        stop_evt = threading.Event()
        upstream_started = threading.Event()
        upstream_closed = threading.Event()

        def accept_loop() -> None:
            while not stop_evt.is_set():
                try:
                    c, _ = srv.accept()
                except OSError:
                    break

                def handle_one(sock: socket.socket) -> None:
                    try:
                        sock.settimeout(5.0)
                        req = sock.recv(4096)
                        if req.startswith(b"GET /v1/models"):
                            body = json.dumps(
                                {"object": "list", "data": [{"id": "m-active-cancel"}]}
                            ).encode("utf-8")
                            sock.sendall(
                                b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: application/json\r\n"
                                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                                + b"Connection: close\r\n\r\n"
                                + body
                            )
                            return
                        # Simulate long prefill/generation: hold open until router closes socket.
                        upstream_started.set()
                        while not stop_evt.is_set():
                            data = sock.recv(1)
                            if not data:
                                upstream_closed.set()
                                break
                    except OSError:
                        upstream_closed.set()
                    finally:
                        try:
                            sock.close()
                        except OSError:
                            pass

                threading.Thread(target=handle_one, args=(c,), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        backup = self._track(_MockOpenAIBackend("cancel-backup", ["m-active-cancel"]))
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                backends=[
                    BackendConfig(
                        id="slow-cancel-primary",
                        local=True,
                        base_url=f"http://127.0.0.1:{port}/v1",
                        timeout_s=5.0,
                    ),
                    BackendConfig(
                        id="cancel-backup",
                        local=False,
                        base_url=backup.base_url,
                        timeout_s=2.0,
                    ),
                ],
            )
            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            req_body = json.dumps(
                {
                    "model": "m-active-cancel",
                    "messages": [{"role": "user", "content": "cancel mid-prefill"}],
                }
            ).encode("utf-8")
            sock = socket.create_connection(("127.0.0.1", rport), timeout=2.0)
            try:
                sock.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    + f"Host: 127.0.0.1:{rport}\r\n".encode("ascii")
                    + b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(req_body)}\r\n".encode("ascii")
                    + b"Connection: close\r\n\r\n"
                    + req_body
                )
                self.assertTrue(upstream_started.wait(timeout=2.0))
            finally:
                t_close = time.monotonic()
                _abort_tcp_socket(sock)

            # Active upstream connection must be closed promptly well before the 5.0s timeout.
            self.assertTrue(upstream_closed.wait(timeout=1.5))
            self.assertLess(time.monotonic() - t_close, 1.5)

            # Wait briefly for handler finally block to release slot.
            dl = time.monotonic() + 1.0
            while router.engine._backends[0].inflight > 0 and time.monotonic() < dl:
                time.sleep(0.01)
            self.assertEqual(router.engine._backends[0].inflight, 0)
            self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
            self.assertTrue(router.engine._backends[0].healthy)
            self.assertEqual(len(backup.requests_seen), 0)
        finally:
            stop_evt.set()
            srv.close()

    def test_sse_buffers_charged_to_shared_response_memory_budget_and_released(
        self,
    ) -> None:
        first_tok = json.dumps(
            {
                "id": "chatcmpl-ssemem",
                "object": "chat.completion.chunk",
                "model": "m-ssemem",
                "choices": [
                    {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}
                ],
            }
        )
        # 1. Pre-commit SSE buffer exceeding max_total_response_memory_bytes -> 503 without cooldown.
        primary = self._track(
            _MockOpenAIBackend(
                "sse-mem-primary",
                ["m-ssemem"],
                raw_sse_bytes=(b": " + (b"k" * 100) + b"\n") * 20,
            )
        )
        backup = self._track(_MockOpenAIBackend("sse-mem-backup", ["m-ssemem"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            max_sse_pre_commit_bytes=65536,
            max_total_response_memory_bytes=1024,
            backends=[
                BackendConfig(
                    id="sse-mem-primary",
                    local=True,
                    base_url=primary.base_url,
                ),
                BackendConfig(
                    id="sse-mem-backup",
                    local=False,
                    base_url=backup.base_url,
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        st_pre, hdrs_pre, bd_pre = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ssemem",
                "stream": True,
                "messages": [{"role": "user", "content": "pre-commit mem cap"}],
            },
        )
        self.assertEqual(st_pre, 503)
        self.assertIn("server_memory_limit_reached", bd_pre.decode("utf-8"))
        self.assertEqual(hdrs_pre["x-freetpu-failovers"], "0")
        self.assertEqual(len(backup.requests_seen), 0)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
        self.assertTrue(router.engine._backends[0].healthy)

        # 2. Post-commit SSE event exceeding shared response memory budget aborts without cooldown.
        primary.raw_sse_bytes = (
            f"data: {first_tok}\n\n".encode("utf-8")
            + (b"data: " + (b"X" * 100) + b"\n") * 20
        )
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "model": "m-ssemem",
                        "stream": True,
                        "messages": [{"role": "user", "content": "post-commit mem cap"}],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            partial = exc_info.exception.partial.decode("utf-8", errors="replace")
            self.assertIn("Hi", partial)
            self.assertIn("server_memory_limit_reached", partial)
        finally:
            conn.close()
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
        self.assertTrue(router.engine._backends[0].healthy)

        # 3. Normal SSE stream within budget succeeds and releases 100% of reserved memory.
        primary.raw_sse_bytes = (
            f"data: {first_tok}\n\ndata: [DONE]\n\n".encode("utf-8")
        )
        st_ok, _, bd_ok = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ssemem",
                "stream": True,
                "messages": [{"role": "user", "content": "normal sse"}],
            },
        )
        self.assertEqual(st_ok, 200)
        self.assertIn("Hi", bd_ok.decode("utf-8"))
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

    def test_explicit_configured_listener_hostname_preserved_in_trusted_hosts(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("bind-host-b", ["m-bindhost"]))
        cfg = RouterConfig(
            host="localhost",
            port=0,
            cors_allowed_origins=[],
            backends=[
                BackendConfig(
                    id="bind-host-b",
                    base_url=backend.base_url,
                    models=["m-bindhost"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        # FreeTPURouterServer preserved configured_bind_host before HTTPServer resolved it to numeric IP.
        self.assertEqual(router.configured_bind_host, "localhost")

        # Simulate binding to a resolvable non-loopback machine hostname (e.g. "worker-node.local").
        router.configured_bind_host = "worker-node.local"

        st_get, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Host": f"worker-node.local:{rport}"},
        )
        self.assertEqual(st_get, 200)

        st_post, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={
                "Host": f"worker-node.local:{rport}",
                "Origin": f"http://worker-node.local:{rport}",
            },
            body={"model": "m-bindhost", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(st_post, 200)

        # Wrong port and unrelated hostname are still rejected with 403.
        wrong_port = rport + 1 if rport < 65000 else rport - 1
        st_wp, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Host": f"worker-node.local:{wrong_port}"},
        )
        self.assertEqual(st_wp, 403)

        st_evil, _, _ = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Host": f"evil.example:{rport}"},
        )
        self.assertEqual(st_evil, 403)

    def test_sse_active_inflight_zero_after_all_termination_paths(self) -> None:
        def wait_inflight_zero(router: FreeTPURouterServer, timeout_s: float = 1.5) -> None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                _, health = router.engine.health_payload()
                if health["active_inflight"] == 0 and all(
                    b["inflight"] == 0 for b in health["backends"]
                ):
                    break
                time.sleep(0.01)
            _, health = router.engine.health_payload()
            self.assertEqual(health["active_inflight"], 0)
            for b in health["backends"]:
                self.assertEqual(b["inflight"], 0)
            self.assertEqual(router.engine._resp_mem_used_bytes, 0)
            _, _, metrics_txt = _http_request(
                router.server_address[1], "GET", "/metrics"
            )
            self.assertIn(
                'freetpu_router_active_inflight{backend_id="leak-b"} 0',
                metrics_txt.decode("utf-8"),
            )

        backend = self._track(
            _MockOpenAIBackend("leak-b", ["m-leak"], chunked_keepalive_sse=True)
        )
        hold_ev = threading.Event()
        backend.keepalive_hold_event = hold_ev
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            request_timeout_s=0.4,
            failure_cooldown_s=0.05,
            backends=[
                BackendConfig(
                    id="leak-b",
                    base_url=backend.base_url,
                    models=["m-leak"],
                    timeout_s=0.4,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Normal SSE completion while upstream keeps TCP socket open (keep-alive)
        st, _, body = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-leak",
                "stream": True,
                "messages": [{"role": "user", "content": "normal keepalive"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn("data: [DONE]", body.decode("utf-8"))
        hold_ev.set()
        wait_inflight_zero(router)

        # 2. Client mid-stream disconnect after reading first chunk
        backend.chunked_keepalive_sse = False
        backend.post_token_keepalive_interval_s = 0.02
        req_body = json.dumps(
            {
                "model": "m-leak",
                "stream": True,
                "messages": [{"role": "user", "content": "mid cancel"}],
            }
        ).encode("utf-8")
        cli = socket.create_connection(("127.0.0.1", rport), timeout=2.0)
        try:
            cli.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                + f"Host: 127.0.0.1:{rport}\r\n".encode("ascii")
                + b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(req_body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + req_body
            )
            chunk = cli.recv(4096)
            self.assertIn(b"200 OK", chunk)
        finally:
            cli.close()
        wait_inflight_zero(router)

        # 3. Post-commit keep-alive stall timeout
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=req_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead):
                resp.read()
        finally:
            conn.close()
        wait_inflight_zero(router)

        # 4. Upstream premature abort before first token -> 502
        abort_backend = self._track(
            _MockOpenAIBackend(
                "leak-abort-b",
                ["m-leak-abort"],
                abort_stream_before_first_token=True,
            )
        )
        cfg_abort = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            request_timeout_s=0.4,
            failure_cooldown_s=0.05,
            backends=[
                BackendConfig(
                    id="leak-abort-b",
                    base_url=abort_backend.base_url,
                    models=["m-leak-abort"],
                    timeout_s=0.4,
                )
            ],
        )
        router_abort = self._track(create_router_server(cfg_abort))
        st_abort, _, _ = _http_request(
            router_abort.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-leak-abort",
                "stream": True,
                "messages": [{"role": "user", "content": "abort"}],
            },
        )
        self.assertEqual(st_abort, 502)
        self.assertEqual(router_abort.engine.health_payload()[1]["active_inflight"], 0)
        self.assertEqual(router_abort.engine._backends[0].inflight, 0)

        # 5. Lease garbage collection safety net
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        router.engine._backends[0].healthy = True
        lease_res = router.engine.reserve_backend_lease(router.engine._backends[0])
        self.assertIsNotNone(lease_res)
        self.assertEqual(router.engine.metrics.total_inflight(), 1)
        del lease_res
        gc.collect()
        wait_inflight_zero(router)

        # 6. Concurrent mixed batch (normal + client mid-stream cancel)
        ok_backend = self._track(_MockOpenAIBackend("leak-mix-b", ["m-leak-mix"]))
        cfg_mix = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="leak-mix-b",
                    base_url=ok_backend.base_url,
                    models=["m-leak-mix"],
                )
            ],
        )
        router_mix = self._track(create_router_server(cfg_mix))
        mport = router_mix.server_address[1]
        mix_body = json.dumps(
            {
                "model": "m-leak-mix",
                "stream": True,
                "messages": [{"role": "user", "content": "mix"}],
            }
        ).encode("utf-8")
        errors: List[Exception] = []

        def run_ok() -> None:
            try:
                s, _, bd = _http_request(
                    mport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-leak-mix",
                        "stream": True,
                        "messages": [{"role": "user", "content": "ok"}],
                    },
                )
                if s != 200 or b"[DONE]" not in bd:
                    errors.append(AssertionError(f"Unexpected ok response {s}"))
            except Exception as exc:
                errors.append(exc)

        def run_cancel() -> None:
            try:
                c = socket.create_connection(("127.0.0.1", mport), timeout=2.0)
                c.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    + f"Host: 127.0.0.1:{mport}\r\n".encode("ascii")
                    + b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(mix_body)}\r\n".encode("ascii")
                    + b"Connection: close\r\n\r\n"
                    + mix_body
                )
                c.recv(512)
                c.close()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_ok if i % 2 == 0 else run_cancel, daemon=True)
            for i in range(24)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)
        self.assertEqual(errors, [])
        dl = time.monotonic() + 1.5
        while router_mix.engine.health_payload()[1]["active_inflight"] > 0 and time.monotonic() < dl:
            time.sleep(0.01)
        self.assertEqual(router_mix.engine.health_payload()[1]["active_inflight"], 0)
        self.assertEqual(router_mix.engine._backends[0].inflight, 0)
        self.assertEqual(router_mix.engine._resp_mem_used_bytes, 0)

    def test_sse_concurrency_scaling_c1_to_c128_overhead_and_no_leak(self) -> None:
        num_tokens = 128
        upstream_script = """
import asyncio, json, socket, sys

TOKENS = int(sys.argv[1])
CHUNKS = [
    ("data: " + json.dumps({"id": "chatcmpl-bench", "object": "chat.completion.chunk", "model": "m-bench", "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}) + "\\n\\n").encode()
    for i in range(TOKENS - 1)
] + [
    ("data: " + json.dumps({"id": "chatcmpl-bench", "object": "chat.completion.chunk", "model": "m-bench", "choices": [{"index": 0, "delta": {"content": "end"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": TOKENS}}) + "\\n\\n").encode(),
    b"data: [DONE]\\n\\n",
]

async def handle(reader, writer):
    sock = writer.get_extra_info("socket")
    if sock is not None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        hdr = await reader.readuntil(b"\\r\\n\\r\\n")
        cl = 0
        for line in hdr.decode("latin1").split("\\r\\n"):
            if line.lower().startswith("content-length:"):
                cl = int(line.split(":", 1)[1].strip())
        if cl:
            await reader.readexactly(cl)
        if b"GET /v1/models" in hdr:
            body = b'{"object":"list","data":[{"id":"m-bench","object":"model"}]}'
            writer.write(
                b"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\n"
                + f"Content-Length: {len(body)}\\r\\nConnection: close\\r\\n\\r\\n".encode()
                + body
            )
            await writer.drain()
            return
        writer.write(
            b"HTTP/1.1 200 OK\\r\\nContent-Type: text/event-stream\\r\\n"
            b"Cache-Control: no-cache\\r\\nConnection: close\\r\\n\\r\\n"
        )
        await writer.drain()
        for chunk in CHUNKS:
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.002)
    except Exception:
        pass
    finally:
        writer.close()

async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", 0, backlog=1024)
    print(srv.sockets[0].getsockname()[1], flush=True)
    async with srv:
        await srv.serve_forever()

asyncio.run(main())
"""
        up_proc = subprocess.Popen(
            [sys.executable, "-c", upstream_script, str(num_tokens)],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert up_proc.stdout is not None
            up_port = int(up_proc.stdout.readline().strip())
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="lowest_latency",
                max_concurrent_connections=512,
                queue_pressure_threshold=256,
                backends=[
                    BackendConfig(
                        id="bench-upstream",
                        base_url=f"http://127.0.0.1:{up_port}/v1",
                        models=["m-bench"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            req_body = json.dumps(
                {
                    "model": "m-bench",
                    "stream": True,
                    "messages": [{"role": "user", "content": "benchmark"}],
                }
            ).encode("utf-8")

            def make_raw_req(port: int) -> bytes:
                return (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(req_body)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii") + req_body

            async def stream_one(port: int, raw_req: bytes) -> Tuple[int, bool]:
                r, w = await asyncio.open_connection("127.0.0.1", port)
                try:
                    sock = w.get_extra_info("socket")
                    if sock is not None:
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    w.write(raw_req)
                    await w.drain()
                    await r.readuntil(b"\r\n\r\n")
                    chunks = 0
                    saw_done = False
                    while True:
                        line = await r.readline()
                        if not line:
                            break
                        if line.startswith(b"data: "):
                            if line.strip() == b"data: [DONE]":
                                saw_done = True
                            else:
                                chunks += 1
                    return chunks, saw_done
                finally:
                    w.close()
                    try:
                        await w.wait_closed()
                    except (OSError, asyncio.CancelledError):
                        pass

            async def run_batch(
                port: int,
                concurrency: int,
                *,
                batch_timeout_s: float = 15.0,
            ) -> float:
                raw_req = make_raw_req(port)
                t0 = time.perf_counter()
                tasks = [
                    asyncio.create_task(stream_one(port, raw_req))
                    for _ in range(concurrency)
                ]
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks),
                        timeout=batch_timeout_s,
                    )
                except asyncio.TimeoutError:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self.fail(
                        f"SSE concurrency batch timed out after {batch_timeout_s:.1f}s "
                        f"(port={port}, concurrency={concurrency})"
                    )
                wall_ms = (time.perf_counter() - t0) * 1000.0
                for chunks, saw_done in results:
                    self.assertEqual(chunks, num_tokens)
                    self.assertTrue(saw_done)
                return wall_ms

            # Warmup
            asyncio.run(run_batch(up_port, 1))
            asyncio.run(run_batch(rport, 1))

            soft_nofile: Optional[int] = None
            if resource is not None and hasattr(resource, "RLIMIT_NOFILE"):
                try:
                    soft_nofile, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                except (OSError, ValueError):
                    soft_nofile = None

            tested_levels = 0
            for concurrency in (1, 8, 16, 64, 128):
                # Each live stream uses ~3 FDs in the test process (client, accepted, upstream).
                if (
                    soft_nofile is not None
                    and soft_nofile > 0
                    and (3 * concurrency + 80) > soft_nofile
                ):
                    continue
                tested_levels += 1
                direct_ms = asyncio.run(run_batch(up_port, concurrency))
                router_ms = asyncio.run(run_batch(rport, concurrency))
                total_chunks = float(concurrency * num_tokens)
                per_chunk_overhead_ms = (router_ms - direct_ms) / total_chunks
                self.assertLess(
                    per_chunk_overhead_ms,
                    0.5,
                    f"c={concurrency} overhead {per_chunk_overhead_ms:.3f} ms/chunk "
                    f"(router={router_ms:.1f}ms, direct={direct_ms:.1f}ms)",
                )
                _, health = router.engine.health_payload()
                self.assertEqual(health["active_inflight"], 0)
                self.assertEqual(health["backends"][0]["inflight"], 0)
                self.assertEqual(router.engine._resp_mem_used_bytes, 0)
            self.assertGreater(tested_levels, 0)
        finally:
            if up_proc.stdout is not None:
                up_proc.stdout.close()
            up_proc.terminate()
            up_proc.wait(timeout=3)

    def test_slow_sse_client_does_not_block_other_streams_and_evicts_cleanly(self) -> None:
        stop_srv = threading.Event()
        upstream_closed_event = threading.Event()
        upstream_closed_at: List[float] = []
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.bind(("127.0.0.1", 0))
        srv_sock.listen(64)
        srv_sock.settimeout(0.2)
        up_port = srv_sock.getsockname()[1]

        first_slow = (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-slow",
                    "object": "chat.completion.chunk",
                    "model": "m-slow",
                    "choices": [
                        {"index": 0, "delta": {"content": "init"}, "finish_reason": None}
                    ],
                }
            )
            + "\n\n"
        ).encode("utf-8")
        big_slow_chunk = (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-slow",
                    "object": "chat.completion.chunk",
                    "model": "m-slow",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "S" * 32768},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            + "\n\n"
        ).encode("utf-8")
        fast_chunks = [
            (
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-fast",
                        "object": "chat.completion.chunk",
                        "model": "m-fast",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": f"ok{i}"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            for i in range(3)
        ] + [b"data: [DONE]\n\n"]

        def handle_conn(conn: socket.socket) -> None:
            is_slow = False
            try:
                conn.settimeout(3.0)
                buf = b""
                while b"\r\n\r\n" not in buf:
                    part = conn.recv(4096)
                    if not part:
                        return
                    buf += part
                hdr_raw, rest = buf.split(b"\r\n\r\n", 1)
                hdr_txt = hdr_raw.decode("latin1")
                if hdr_txt.startswith("GET /v1/models"):
                    body = json.dumps(
                        {
                            "object": "list",
                            "data": [
                                {"id": "m-slow", "object": "model"},
                                {"id": "m-fast", "object": "model"},
                            ],
                        }
                    ).encode("utf-8")
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                        + body
                    )
                    return
                cl = 0
                for line in hdr_txt.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        cl = int(line.split(":", 1)[1].strip())
                while len(rest) < cl:
                    part = conn.recv(cl - len(rest))
                    if not part:
                        break
                    rest += part
                req_obj = json.loads(rest[:cl].decode("utf-8"))
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
                )
                if req_obj.get("model") == "m-slow":
                    is_slow = True
                    conn.sendall(first_slow)
                    # Keep streaming indefinitely without [DONE] until the router evicts
                    # the stalled client and closes this upstream connection.
                    conn.settimeout(0.25)
                    while not stop_srv.is_set():
                        view = memoryview(big_slow_chunk)
                        offset = 0
                        while offset < len(view) and not stop_srv.is_set():
                            try:
                                sent = conn.send(view[offset:])
                            except TimeoutError:
                                continue
                            if sent <= 0:
                                raise BrokenPipeError("upstream peer closed")
                            offset += sent
                else:
                    for ch in fast_chunks:
                        conn.sendall(ch)
            except OSError:
                if is_slow and not stop_srv.is_set():
                    upstream_closed_at.append(time.monotonic())
                    upstream_closed_event.set()
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def accept_loop() -> None:
            while not stop_srv.is_set():
                try:
                    c, _ = srv_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=handle_conn, args=(c,), daemon=True).start()

        accept_thread = threading.Thread(target=accept_loop, daemon=True)
        accept_thread.start()
        slow_sock: Optional[socket.socket] = None
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="lowest_latency",
                request_timeout_s=5.0,
                client_read_timeout_s=0.6,
                backends=[
                    BackendConfig(
                        id="sse-backpressure-b",
                        base_url=f"http://127.0.0.1:{up_port}/v1",
                        models=["m-slow", "m-fast"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
            rport = router.server_address[1]

            # Set a small SO_RCVBUF BEFORE connect() so both Linux and macOS advertise
            # a small TCP receive window and backpressure fills quickly inside _SSEReactor.
            slow_body = json.dumps(
                {
                    "model": "m-slow",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stall me"}],
                }
            ).encode("utf-8")
            slow_start = time.monotonic()
            slow_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            slow_sock.settimeout(3.0)
            try:
                slow_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            except OSError:
                pass
            slow_sock.connect(("127.0.0.1", rport))
            slow_sock.sendall(
                (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{rport}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(slow_body)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii")
                + slow_body
            )
            slow_hdr = b""
            while b"\r\n\r\n" not in slow_hdr:
                part = slow_sock.recv(1024)
                if not part:
                    break
                slow_hdr += part
            self.assertIn(b"200 OK", slow_hdr)

            # Give the reactor time to saturate the stalled client's socket send buffer.
            time.sleep(0.15)

            # Fast SSE streams on the SAME shared reactor must complete promptly without stalling.
            t0 = time.monotonic()
            for _ in range(3):
                st, _, bd = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-fast",
                        "stream": True,
                        "messages": [{"role": "user", "content": "fast"}],
                    },
                )
                self.assertEqual(st, 200)
                self.assertIn(b"data: [DONE]", bd)
            fast_elapsed = time.monotonic() - t0
            self.assertLess(fast_elapsed, 1.0)

            # The slow upstream MUST observe socket close due to write-timeout eviction
            # well BEFORE the overall 5.0s request_timeout_s (and without ever sending [DONE]).
            self.assertTrue(
                upstream_closed_event.wait(timeout=2.5),
                "Slow upstream was not closed by router write-timeout eviction",
            )
            self.assertTrue(upstream_closed_at)
            self.assertLess(upstream_closed_at[0] - slow_start, 2.5)

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                _, health = router.engine.health_payload()
                if health["active_inflight"] == 0 and router.engine._resp_mem_used_bytes == 0:
                    break
                time.sleep(0.05)
            _, health = router.engine.health_payload()
            self.assertEqual(health["active_inflight"], 0)
            self.assertEqual(router.engine._resp_mem_used_bytes, 0)
            self.assertTrue(router.engine._backends[0].healthy)
            self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
            slow_events = [e for e in recorded_usage if e.model == "m-slow"]
            self.assertEqual(len(slow_events), 1)
            self.assertEqual(slow_events[0].status_code, 499)
            self.assertGreater(slow_events[0].completion_tokens, 0)
        finally:
            stop_srv.set()
            if slow_sock is not None:
                try:
                    slow_sock.close()
                except OSError:
                    pass
            try:
                srv_sock.close()
            except OSError:
                pass
            accept_thread.join(timeout=2.0)

    def test_streaming_usage_fallback_uses_generated_content_not_raw_json_length(
        self,
    ) -> None:
        recorded: List[UsageEvent] = []

        class _CaptureUsageHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded.append(event)

        huge_id = "chatcmpl-" + ("x" * 2048)
        huge_fp = "fp_" + ("z" * 1024)
        huge_logprobs = {
            "content": [
                {
                    "token": "Hi",
                    "logprob": -0.05,
                    "bytes": [72, 105],
                    "top_logprobs": [{"token": "Hello", "logprob": -1.5}] * 16,
                }
            ]
        }
        # Exact generated characters: "Hi" (2) + " there" (6) + "plan" (4) + "get" (3) + "{}" (2) = 17 chars -> 5 tokens.
        sse_chunks = [
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000001,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "logprobs": huge_logprobs,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000002,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000003,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000004,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " there"},
                        "logprobs": huge_logprobs,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000005,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "plan"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000006,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": huge_id,
                "object": "chat.completion.chunk",
                "created": 1700000007,
                "model": "m-usage-est",
                "system_fingerprint": huge_fp,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        backend = self._track(
            _MockOpenAIBackend(
                "usage-est-b",
                ["m-usage-est"],
                custom_sse_chunks=sse_chunks,
            )
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="usage-est-b",
                    base_url=backend.base_url,
                    models=["m-usage-est"],
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=2.0,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureUsageHook()))
        rport = router.server_address[1]

        st, _, bd = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-usage-est",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(b"data: [DONE]", bd)
        self.assertEqual(len(recorded), 1)
        ev = recorded[0]
        self.assertEqual(ev.completion_tokens, 5)
        expected_cost = (ev.prompt_tokens * 1.0 + 5 * 2.0) / 1_000_000.0
        self.assertAlmostEqual(ev.estimated_cost_usd, expected_cost, places=12)

        # Also verify `/v1/completions` streaming with `choice.text` + huge metadata.
        recorded.clear()
        backend.custom_sse_chunks = [
            {
                "id": huge_id,
                "object": "text_completion",
                "created": 1700000010,
                "model": "m-usage-est",
                "choices": [{"index": 0, "text": "abcd", "finish_reason": None}],
            },
            {
                "id": huge_id,
                "object": "text_completion",
                "created": 1700000011,
                "model": "m-usage-est",
                "choices": [{"index": 0, "text": "efgh", "finish_reason": "stop"}],
            },
        ]
        st2, _, bd2 = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={"model": "m-usage-est", "stream": True, "prompt": "go"},
        )
        self.assertEqual(st2, 200)
        self.assertIn(b"data: [DONE]", bd2)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].completion_tokens, 2)

    def test_high_fd_client_preserves_failover_without_select_select(self) -> None:
        primary = self._track(
            _MockOpenAIBackend("highfd-primary", ["m-highfd"], status_code=200)
        )
        backup = self._track(_MockOpenAIBackend("highfd-backup", ["m-highfd"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="highfd-primary",
                    local=True,
                    base_url=primary.base_url,
                    models=["m-highfd"],
                ),
                BackendConfig(
                    id="highfd-backup",
                    local=False,
                    base_url=backup.base_url,
                    models=["m-highfd"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        # Flip primary to 503 after initial health probe so the request hits primary first
        # and exercises _is_client_disconnected() before failing over to backup.
        primary.status_code = 503

        def _forbid_select(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("filedescriptor out of range in select()")

        patch_ctx = (
            mock.patch.object(select, "select", side_effect=_forbid_select)
            if hasattr(select, "poll")
            else nullcontext()
        )
        with patch_ctx:
            st, hdrs, bd = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-highfd",
                    "messages": [{"role": "user", "content": "failover"}],
                },
            )
        self.assertEqual(st, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "highfd-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn(b"highfd-backup", bd)

    def test_tpu_v6e8_playbook_uses_supported_jax_and_vllm_cache_vars(self) -> None:
        playbook = (
            REPO_ROOT / "docs/router/vllm-tpu-v6e8-batch-scaling.md"
        ).read_text(encoding="utf-8")
        router_readme = (REPO_ROOT / "docs/router/README.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("XLA_COMPILATION_CACHE_DIR", playbook)
        self.assertNotIn("local NVMe", playbook)
        self.assertNotIn("Load-Time MXFP4", playbook)
        self.assertIn("Hyperdisk Balanced", playbook)
        self.assertIn("hyperdisk-balanced", playbook)
        self.assertNotIn("Retained Persistent Disk", playbook)
        self.assertIn("gs://", playbook)
        self.assertNotIn("/var/cache/freetpu-xla-cache", playbook)
        self.assertIn("export XLA_CACHE_DIR=/mnt/disks/tpu-cache/xla", playbook)
        self.assertIn("export JAX_COMPILATION_CACHE_DIR=/xla-cache", playbook)
        self.assertIn("export VLLM_XLA_CACHE_PATH=/xla-cache", playbook)
        self.assertIn("/mnt/disks/tpu-cache/xla}:/xla-cache", playbook)
        self.assertIn("-e JAX_COMPILATION_CACHE_DIR=", playbook)
        self.assertIn("-e VLLM_XLA_CACHE_PATH=", playbook)
        self.assertIn("-e LIBTPU_INIT_ARGS=", playbook)
        self.assertIn("-e HF_HUB_CACHE=/hf-cache", playbook)
        self.assertIn("/mnt/disks/tpu-cache/hf", playbook)
        self.assertIn("Qwen/Qwen3-32B", playbook)
        self.assertIn("unsloth/Llama-3.3-70B-Instruct", playbook)
        self.assertNotIn("Preset C", playbook)
        self.assertNotIn("compressed-tensors", playbook)
        self.assertNotIn("W8A8", playbook)
        self.assertNotIn("W8A16", playbook)
        self.assertNotIn("W8A8", router_readme)
        self.assertNotIn("W8A16", router_readme)
        self.assertNotIn("gpt-oss-120b", playbook)
        self.assertIn("ops/tpu-exporter/tpu_exporter.py", playbook)
        self.assertIn("127.0.0.1:9400/metrics", playbook)
        self.assertLess(
            playbook.index("ops/tpu-exporter/tpu_exporter.py"),
            playbook.index("python3 router/freetpu_router.py"),
        )
        self.assertIn("single attention request bucket", playbook.lower())
        self.assertIn("power-of-two", playbook.lower())
        self.assertIn("16, 32, 64, 128, 256, 512, 1024, 2048", playbook)
        self.assertNotIn("256-aligned", playbook)
        self.assertNotIn("vllm/vllm-tpu:latest", playbook)
        self.assertIn("vllm/vllm-tpu:v0.33.0", playbook)
        self.assertIn(
            """--additional-config '{"enable_continue_decode": true}'""",
            playbook,
        )
        self.assertIn("arbitrary code execution", playbook.lower())
        self.assertIn("write iam", playbook.lower())
        self.assertIn("trust domains", playbook.lower())

    def test_deadline_socket_waits_support_high_fds_without_select_select(self) -> None:
        def _forbid_select(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("filedescriptor out of range in select()")

        def _maybe_forbid_select() -> Any:
            return (
                mock.patch.object(select, "select", side_effect=_forbid_select)
                if hasattr(select, "poll")
                else nullcontext()
            )

        # 1. Upstream-only wait path (client_sock=None) works without select.select() on poll platforms.
        up_r, up_w = socket.socketpair()
        try:
            def _delayed_send() -> None:
                time.sleep(0.05)
                up_w.sendall(b"hello")

            threading.Thread(target=_delayed_send, daemon=True).start()
            with _maybe_forbid_select():
                ds = DeadlineSocket(up_r, time.monotonic() + 1.0, label="test-up")
                fp = ds.makefile("rb")
                try:
                    self.assertEqual(fp.read(5), b"hello")
                finally:
                    fp.close()
        finally:
            up_r.close()
            up_w.close()

        # 2. Client-watch wait path works for abrupt client disconnect (TCP RST).
        up_r2, up_w2 = socket.socketpair()
        cli_r2, cli_w2 = _tcp_socketpair()
        try:
            def _abort_client() -> None:
                time.sleep(0.05)
                _abort_tcp_socket(cli_w2)

            threading.Thread(target=_abort_client, daemon=True).start()
            with _maybe_forbid_select():
                ds2 = DeadlineSocket(
                    up_r2,
                    time.monotonic() + 1.0,
                    label="test-watch",
                    client_sock=cli_r2,
                )
                fp2 = ds2.makefile("rb")
                try:
                    with self.assertRaises(ClientDisconnectedError):
                        fp2.read(5)
                finally:
                    fp2.close()
        finally:
            up_r2.close()
            up_w2.close()
            cli_r2.close()
            cli_w2.close()

        # 3. Exercise the SelectSelector fallback path explicitly (Windows path without select.poll).
        import selectors as selectors_mod
        import router.tpu_telemetry as tpu_telemetry_mod

        sel_r, sel_w = socket.socketpair()
        try:
            def _delayed_sel_send() -> None:
                time.sleep(0.02)
                sel_w.sendall(b"win")

            threading.Thread(target=_delayed_sel_send, daemon=True).start()
            with mock.patch.object(
                tpu_telemetry_mod, "select", object()
            ), mock.patch.object(
                tpu_telemetry_mod.selectors,
                "DefaultSelector",
                selectors_mod.SelectSelector,
            ):
                ds_sel = DeadlineSocket(sel_r, time.monotonic() + 1.0, label="test-sel")
                fp_sel = ds_sel.makefile("rb")
                try:
                    self.assertEqual(fp_sel.read(3), b"win")
                finally:
                    fp_sel.close()
        finally:
            sel_r.close()
            sel_w.close()

        # 4. Real OS FD >= 1024 test when RLIMIT_NOFILE permits.
        soft_lim = 0
        if resource is not None and hasattr(resource, "RLIMIT_NOFILE"):
            try:
                soft_lim, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            except (OSError, ValueError):
                soft_lim = 0
        if soft_lim > 1060 and hasattr(select, "poll"):
            s_r, s_w = socket.socketpair()
            high_fd: Optional[int] = None
            high_sock: Optional[socket.socket] = None
            try:
                for cand_fd in range(1040, min(soft_lim - 4, 1080)):
                    try:
                        os.fstat(cand_fd)
                        continue
                    except OSError:
                        os.dup2(s_r.fileno(), cand_fd)
                        high_fd = cand_fd
                        break
                if high_fd is not None:
                    high_sock = socket.socket(fileno=high_fd)
                    high_fd = None
                    self.assertGreaterEqual(high_sock.fileno(), 1024)
                    s_w.sendall(b"ok")
                    readable, errored = wait_sockets_readable([high_sock], 0.5)
                    self.assertIn(high_sock, readable)
                    self.assertFalse(errored)
                    self.assertEqual(high_sock.recv(2), b"ok")
            finally:
                if high_sock is not None:
                    high_sock.close()
                if high_fd is not None:
                    os.close(high_fd)
                s_r.close()
                s_w.close()

    def test_scrape_tpu_exporter_enforces_deadline_across_multi_addr_and_tls(
        self,
    ) -> None:
        # Part A: multi-address hostname where each candidate stalls must not multiply timeout.
        recorded_timeouts: List[Optional[float]] = []

        class _StallingConnectSocket:
            def __init__(self, af: int, stype: int, proto: int) -> None:
                self._timeout: Optional[float] = None

            def settimeout(self, t: Optional[float]) -> None:
                self._timeout = t
                recorded_timeouts.append(t)

            def connect(self, sa: Any) -> None:
                sleep_s = min(0.18, self._timeout if self._timeout is not None else 0.18)
                time.sleep(max(0.0, sleep_s))
                raise TimeoutError("simulated connect stall")

            def close(self) -> None:
                pass

        fake_infos = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.1", 9400)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.2", 9400)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.3", 9400)),
        ]
        t0 = time.monotonic()
        with (
            mock.patch.object(socket, "getaddrinfo", return_value=fake_infos),
            mock.patch.object(socket, "socket", side_effect=_StallingConnectSocket),
        ):
            telem = scrape_tpu_exporter(
                "http://multi.tpu.test:9400/metrics", timeout_s=0.30
            )
        elapsed = time.monotonic() - t0
        self.assertFalse(telem.reachable)
        self.assertEqual(telem.error, "exporter_scrape_timeout")
        self.assertLess(elapsed, 0.55)
        self.assertGreaterEqual(len(recorded_timeouts), 2)
        self.assertLess(recorded_timeouts[1], recorded_timeouts[0])

        # Part B: HTTPS server that accepts TCP and stalls during TLS handshake.
        tls_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tls_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tls_srv.bind(("127.0.0.1", 0))
        tls_srv.listen(4)
        tls_port = tls_srv.getsockname()[1]
        accepted: List[socket.socket] = []

        def _stall_tls() -> None:
            try:
                c, _ = tls_srv.accept()
                accepted.append(c)
                time.sleep(2.0)
            except OSError:
                pass

        tls_thread = threading.Thread(target=_stall_tls, daemon=True)
        tls_thread.start()
        try:
            t1 = time.monotonic()
            telem_tls = scrape_tpu_exporter(
                f"https://127.0.0.1:{tls_port}/metrics", timeout_s=0.25
            )
            elapsed_tls = time.monotonic() - t1
            self.assertFalse(telem_tls.reachable)
            self.assertEqual(telem_tls.error, "exporter_scrape_timeout")
            self.assertLess(elapsed_tls, 0.65)
        finally:
            for c in accepted:
                try:
                    c.close()
                except OSError:
                    pass
            tls_srv.close()
            tls_thread.join(timeout=2.5)

    def test_scrape_tpu_exporter_keeps_plain_http_body_socket_nonblocking(self) -> None:
        # Chunked response that delivers its chunk header near the deadline, then stalls payload.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(4)
        port = srv.getsockname()[1]
        accepted: List[socket.socket] = []

        def _serve_stalled_chunk() -> None:
            try:
                c, _ = srv.accept()
                accepted.append(c)
                c.settimeout(1.0)
                buf = b""
                while b"\r\n\r\n" not in buf:
                    part = c.recv(1024)
                    if not part:
                        return
                    buf += part
                c.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/plain\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n\r\n"
                )
                # Send chunk size header near the 0.30s deadline, then stall chunk payload.
                time.sleep(0.20)
                c.sendall(b"a\r\n")
                time.sleep(1.5)
            except OSError:
                pass

        t = threading.Thread(target=_serve_stalled_chunk, daemon=True)
        t.start()
        try:
            t0 = time.monotonic()
            telem = scrape_tpu_exporter(
                f"http://127.0.0.1:{port}/metrics", timeout_s=0.30
            )
            elapsed = time.monotonic() - t0
            self.assertFalse(telem.reachable)
            self.assertEqual(telem.error, "exporter_scrape_timeout")
            self.assertLess(elapsed, 0.50)
        finally:
            for c in accepted:
                try:
                    c.close()
                except OSError:
                    pass
            srv.close()
            t.join(timeout=2.0)

        # Directly confirm _read_bounded_http_body never sets a positive timeout on the raw socket
        # once DeadlineSocket/_DeadlineSocketIO has configured nonblocking mode.
        r_sock, w_sock = socket.socketpair()
        settimeout_calls: List[Optional[float]] = []

        class _SpySocket:
            def __init__(self, inner: socket.socket) -> None:
                self._inner = inner

            def settimeout(self, val: Optional[float]) -> None:
                settimeout_calls.append(val)
                self._inner.settimeout(val)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        try:
            w_sock.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: 5\r\n"
                b"Connection: close\r\n\r\nhello"
            )
            w_sock.close()
            spy = _SpySocket(r_sock)
            ds = DeadlineSocket(spy, time.monotonic() + 1.0, label="spy-body")
            resp = http.client.HTTPResponse(ds)
            resp.begin()
            try:
                body = _read_bounded_http_body(resp, ds, deadline=time.monotonic() + 1.0)
            finally:
                resp.close()
            self.assertEqual(body, b"hello")
            self.assertEqual(settimeout_calls, [0.0])
        finally:
            r_sock.close()

    def test_scrape_tpu_exporter_invalid_url_syntax_returns_unreachable(self) -> None:
        for bad_url in (
            "http://[::1:9100/metrics",
            "http://[bad]host/metrics",
        ):
            telem = scrape_tpu_exporter(bad_url, timeout_s=0.5)
            self.assertFalse(telem.reachable)
            self.assertFalse(telem.exporter_up)
            self.assertIn("Invalid tpu_exporter_url", telem.error or "")

    def test_sanitize_url_strips_query_and_fragment_and_base_url_rejects_them(
        self,
    ) -> None:
        self.assertEqual(
            sanitize_url(
                "https://user:topsecret@api.example.com:8443/v1?subscription-key=supersecret#frag"
            ),
            "https://api.example.com:8443/v1",
        )
        self.assertEqual(
            sanitize_url("http://127.0.0.1:8000/v1?api_key=leakme"),
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            sanitize_url("http://127.0.0.1:8000/v1#secret-fragment"),
            "http://127.0.0.1:8000/v1",
        )
        for bad_base in (
            "https://api.example.com/v1?subscription-key=supersecret",
            "https://api.example.com/v1#supersecret",
            "http://127.0.0.1:8000/v1?",
        ):
            with self.assertRaises(ValueError) as ctx:
                BackendConfig(id="bad-q-b", base_url=bad_base)
            self.assertNotIn("supersecret", str(ctx.exception))

        # Exporter URLs may carry query parameters but must reject fragments.
        ok_exp = validate_http_url(
            "http://127.0.0.1:9400/metrics?target=tpu0",
            field_name="tpu_exporter_url",
            allow_query=True,
        )
        self.assertEqual(ok_exp, "http://127.0.0.1:9400/metrics?target=tpu0")
        with self.assertRaises(ValueError) as ctx_frag:
            BackendConfig(
                id="bad-exp-frag",
                base_url="http://127.0.0.1:8000/v1",
                tpu_exporter_url="http://127.0.0.1:9400/metrics#secretfrag",
            )
        self.assertNotIn("secretfrag", str(ctx_frag.exception))

    def test_shared_request_memory_budget_accounts_for_supplementary_unicode_ucs4(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("ucs4-mem-b", ["m-ucs4"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_total_request_memory_bytes=2000,
            backends=[
                BackendConfig(
                    id="ucs4-mem-b",
                    base_url=backend.base_url,
                    models=["m-ucs4"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Pure ASCII ~300-byte body needs 4x (~1200 bytes) <= 2000 -> succeeds.
        st_ascii, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ucs4",
                "messages": [{"role": "user", "content": "a" * 240}],
            },
        )
        self.assertEqual(st_ascii, 200)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

        # 2. Same-sized body with one raw UTF-8 supplementary char (UCS-4 4x str widening)
        # requires 10x (~3000 bytes) > 2000 total budget -> permanent 413 payload_too_large.
        ucs4_raw = json.dumps(
            {
                "model": "m-ucs4",
                "messages": [{"role": "user", "content": ("a" * 236) + "\U0001f600"}],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=ucs4_raw,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            bd_ucs4 = resp.read()
            self.assertEqual(resp.status, 413)
            self.assertIn(b"payload_too_large", bd_ucs4)
        finally:
            conn.close()
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

        # 3. Same-sized pure-ASCII body with a \uD83D\uDE00 surrogate escape also widens
        # parsed string memory beyond total budget -> permanent 413 payload_too_large.
        esc_raw = (
            '{"model":"m-ucs4","messages":[{"role":"user","content":"'
            + ("a" * 228)
            + '\\uD83D\\uDE00"}]}'
        ).encode("ascii")
        conn2 = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn2.request(
                "POST",
                "/v1/chat/completions",
                body=esc_raw,
                headers={"Content-Type": "application/json"},
            )
            resp2 = conn2.getresponse()
            bd_esc = resp2.read()
            self.assertEqual(resp2.status, 413)
            self.assertIn(b"payload_too_large", bd_esc)
        finally:
            conn2.close()
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_downstream_sse_disconnect_records_499_retains_tokens_and_skips_latency(
        self,
    ) -> None:
        recorded: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded.append(event)

        # Use a raw TCP upstream that sends 1 SSE token then waits for client disconnect
        # so the disconnect happens mid-stream (before [DONE]).
        stop_up = threading.Event()
        up_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        up_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        up_srv.bind(("127.0.0.1", 0))
        up_srv.listen(8)
        up_srv.settimeout(0.2)
        up_port = up_srv.getsockname()[1]

        def _handle_up(conn: socket.socket) -> None:
            try:
                conn.settimeout(2.0)
                buf = b""
                while b"\r\n\r\n" not in buf:
                    p = conn.recv(4096)
                    if not p:
                        return
                    buf += p
                hdr = buf.split(b"\r\n\r\n", 1)[0].decode("latin1")
                if hdr.startswith("GET /v1/models"):
                    body = b'{"object":"list","data":[{"id":"m-disc499","object":"model"}]}'
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                        + body
                    )
                    return
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
                )
                tok = (
                    "data: "
                    + json.dumps(
                        {
                            "id": "chatcmpl-499",
                            "object": "chat.completion.chunk",
                            "model": "m-disc499",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "Hello partial stream"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    + "\n\n"
                ).encode("utf-8")
                conn.sendall(tok)
                stop_up.wait(timeout=3.0)
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def _accept_up() -> None:
            while not stop_up.is_set():
                try:
                    c, _ = up_srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=_handle_up, args=(c,), daemon=True).start()

        up_thread = threading.Thread(target=_accept_up, daemon=True)
        up_thread.start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="disc-499-raw",
                        base_url=f"http://127.0.0.1:{up_port}/v1",
                        models=["m-disc499"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
            rport = router.server_address[1]
            # Reset EWMA latency after probe so we can assert cancelled stream does not record latency.
            router.engine._backends[0].ewma_latency_ms = None

            req_body = json.dumps(
                {
                    "model": "m-disc499",
                    "stream": True,
                    "messages": [{"role": "user", "content": "partial"}],
                }
            ).encode("utf-8")
            cli = socket.create_connection(("127.0.0.1", rport), timeout=3.0)
            try:
                cli.sendall(
                    (
                        f"POST /v1/chat/completions HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{rport}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(req_body)}\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode("ascii")
                    + req_body
                )
                seen = b""
                while b"Hello partial stream" not in seen:
                    p = cli.recv(4096)
                    if not p:
                        break
                    seen += p
                self.assertIn(b"Hello partial stream", seen)
            finally:
                # Client abruptly aborts mid-stream after receiving the first token.
                _abort_tcp_socket(cli)

            deadline = time.monotonic() + 2.0
            while not recorded and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(recorded), 1)
            self.assertEqual(recorded[0].status_code, 499)
            self.assertGreater(recorded[0].completion_tokens, 0)
            self.assertIsNone(router.engine._backends[0].ewma_latency_ms)
            self.assertTrue(router.engine._backends[0].healthy)
            self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

            st_m, _, bd_m = _http_request(rport, "GET", "/metrics")
            self.assertEqual(st_m, 200)
            metrics_txt = bd_m.decode("utf-8")
            self.assertIn('status="499"', metrics_txt)
            self.assertNotIn('model="m-disc499",backend_id="disc-499-raw",status="200"', metrics_txt)
        finally:
            stop_up.set()
            try:
                up_srv.close()
            except OSError:
                pass
            up_thread.join(timeout=2.0)

    def test_scrape_tpu_exporter_enforces_deadline_during_slow_dns_resolution(
        self,
    ) -> None:
        release_dns = threading.Event()

        def _slow_getaddrinfo(*args: Any, **kwargs: Any) -> List[Any]:
            release_dns.wait(timeout=1.5)
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", 9400),
                )
            ]

        t0 = time.monotonic()
        try:
            with mock.patch.object(socket, "getaddrinfo", side_effect=_slow_getaddrinfo):
                telem = scrape_tpu_exporter(
                    "http://slow-dns.tpu.test:9400/metrics", timeout_s=0.25
                )
            elapsed = time.monotonic() - t0
            self.assertFalse(telem.reachable)
            self.assertEqual(telem.error, "exporter_scrape_timeout")
            self.assertLess(elapsed, 0.65)
        finally:
            release_dns.set()

    def test_deadline_socket_cancels_stalled_tls_read_on_client_disconnect(
        self,
    ) -> None:
        for want_exc in (ssl.SSLWantReadError(), ssl.SSLWantWriteError()):
            with self.subTest(want_exc=type(want_exc).__name__):
                up_r, up_w = socket.socketpair()
                cli_r, cli_w = _tcp_socketpair()
                try:
                    class _FakeTLSSocket:
                        """Simulates a nonblocking SSLSocket receiving a partial TLS record."""

                        def __init__(self, inner: socket.socket) -> None:
                            self._inner = inner
                            self._consumed_record = False

                        def fileno(self) -> int:
                            return self._inner.fileno()

                        def settimeout(self, t: Optional[float]) -> None:
                            self._inner.settimeout(t)

                        def pending(self) -> int:
                            return 0

                        def recv_into(self, b: Any) -> int:
                            # Drain the partial TLS record byte so TCP readability clears,
                            # then tell caller OpenSSL needs more read/write I/O.
                            if not self._consumed_record:
                                try:
                                    self._inner.recv(1)
                                except BlockingIOError:
                                    pass
                                self._consumed_record = True
                            raise want_exc

                        def close(self) -> None:
                            self._inner.close()

                    fake_tls = _FakeTLSSocket(up_r)
                    ds = DeadlineSocket(
                        fake_tls,
                        time.monotonic() + 5.0,
                        label="tls-cancel-test",
                        client_sock=cli_r,
                    )
                    ds._is_ssl = True

                    # Make upstream TCP readable (partial TLS record). For SSLWantWriteError,
                    # also saturate up_r's send buffer so POLLOUT blocks until client disconnect.
                    up_w.sendall(b"\x17")
                    if isinstance(want_exc, ssl.SSLWantWriteError):
                        up_r.setblocking(False)
                        try:
                            while True:
                                up_r.send(b"z" * 65536)
                        except BlockingIOError:
                            pass

                    def _disconnect_client() -> None:
                        time.sleep(0.06)
                        _abort_tcp_socket(cli_w)

                    threading.Thread(target=_disconnect_client, daemon=True).start()
                    t0 = time.monotonic()
                    fp = ds.makefile("rb")
                    try:
                        with self.assertRaises(ClientDisconnectedError):
                            fp.read(16)
                    finally:
                        fp.close()
                    self.assertLess(time.monotonic() - t0, 1.0)
                finally:
                    up_r.close()
                    up_w.close()
                    cli_r.close()
                    cli_w.close()

    def test_fixed_length_sse_finishes_immediately_when_buffered_without_done_or_eof(
        self,
    ) -> None:
        stop_up = threading.Event()
        up_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        up_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        up_srv.bind(("127.0.0.1", 0))
        up_srv.listen(8)
        up_srv.settimeout(0.2)
        up_port = up_srv.getsockname()[1]

        sse_body = (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-fixed",
                    "object": "chat.completion.chunk",
                    "model": "m-fixed",
                    "choices": [
                        {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                    ],
                }
            )
            + "\n\n"
            + "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-fixed",
                    "object": "chat.completion.chunk",
                    "model": "m-fixed",
                    "choices": [
                        {"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}
                    ],
                }
            )
            + "\n\n"
        ).encode("utf-8")

        def _handle(conn: socket.socket) -> None:
            try:
                conn.settimeout(2.0)
                buf = b""
                while b"\r\n\r\n" not in buf:
                    p = conn.recv(4096)
                    if not p:
                        return
                    buf += p
                hdr = buf.split(b"\r\n\r\n", 1)[0].decode("latin1")
                if hdr.startswith("GET /v1/models"):
                    body = b'{"object":"list","data":[{"id":"m-fixed","object":"model"}]}'
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
                        + body
                    )
                    return
                # Send fixed-length SSE body with finish_reason="stop" and NO [DONE],
                # then keep the TCP socket open well past request_timeout_s.
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    + f"Content-Length: {len(sse_body)}\r\n\r\n".encode("ascii")
                    + sse_body
                )
                stop_up.wait(timeout=3.0)
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def _accept() -> None:
            while not stop_up.is_set():
                try:
                    c, _ = up_srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=_handle, args=(c,), daemon=True).start()

        up_thread = threading.Thread(target=_accept, daemon=True)
        up_thread.start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                request_timeout_s=1.5,
                backends=[
                    BackendConfig(
                        id="fixed-sse-b",
                        base_url=f"http://127.0.0.1:{up_port}/v1",
                        models=["m-fixed"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            t0 = time.monotonic()
            st, _, bd = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-fixed",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            elapsed = time.monotonic() - t0
            self.assertEqual(st, 200)
            self.assertLess(elapsed, 0.6)
            self.assertIn(b"hello", bd)
            self.assertIn(b" world", bd)
            self.assertNotIn(b"upstream_stream_interrupted", bd)
            _, health = router.engine.health_payload()
            self.assertEqual(health["active_inflight"], 0)
            self.assertTrue(router.engine._backends[0].healthy)
            self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)
        finally:
            stop_up.set()
            try:
                up_srv.close()
            except OSError:
                pass
            up_thread.join(timeout=2.0)

    def test_http_1_0_sse_request_uses_close_delimited_framing_not_chunked(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("http10-sse-b", ["m-http10"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="http10-sse-b",
                    base_url=backend.base_url,
                    models=["m-http10"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        req_body = json.dumps(
            {
                "model": "m-http10",
                "stream": True,
                "messages": [{"role": "user", "content": "hello 1.0"}],
            }
        ).encode("utf-8")
        sock = socket.create_connection(("127.0.0.1", rport), timeout=3.0)
        try:
            sock.sendall(
                (
                    f"POST /v1/chat/completions HTTP/1.0\r\n"
                    f"Host: 127.0.0.1:{rport}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(req_body)}\r\n\r\n"
                ).encode("ascii")
                + req_body
            )
            chunks: List[bytes] = []
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                chunks.append(part)
            raw_resp = b"".join(chunks)
        finally:
            sock.close()

        hdr_raw, body_raw = raw_resp.split(b"\r\n\r\n", 1)
        hdr_lower = hdr_raw.decode("latin1").lower()
        self.assertIn("200 ok", hdr_lower)
        self.assertNotIn("transfer-encoding", hdr_lower)
        self.assertIn("connection: close", hdr_lower)
        self.assertTrue(body_raw.startswith(b"data: "), f"Unexpected chunk framing: {body_raw[:64]!r}")
        self.assertTrue(body_raw.endswith(b"data: [DONE]\n\n"))
        self.assertNotIn(b"\r\n0\r\n\r\n", body_raw)

    def test_prometheus_parser_handles_leading_trailing_dot_and_lowercase_nan(
        self,
    ) -> None:
        # Leading dot (.97) triggers HBM OOM risk (>= 0.95).
        hbm_text = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 1\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} .97\n'
        )
        telem_hbm = summarize_tpu_samples(parse_prometheus_text(hbm_text))
        self.assertAlmostEqual(telem_hbm.max_hbm_utilization_ratio or 0.0, 0.97, places=6)
        ready_hbm, reason_hbm = evaluate_tpu_readiness(telem_hbm, max_hbm_utilization=0.95)
        self.assertFalse(ready_hbm)
        self.assertEqual(reason_hbm, "tpu_hbm_oom_risk")

        # Trailing dot (97.) triggers duty-cycle overload (>= 95.0).
        duty_text = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 1\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
            'freetpu_tpu_duty_cycle_percent{chip="0"} 97.\n'
        )
        telem_duty = summarize_tpu_samples(parse_prometheus_text(duty_text))
        self.assertAlmostEqual(telem_duty.max_duty_cycle_percent or 0.0, 97.0, places=6)
        ready_duty, reason_duty = evaluate_tpu_readiness(
            telem_duty, max_duty_cycle_percent=95.0
        )
        self.assertFalse(ready_duty)
        self.assertEqual(reason_duty, "tpu_duty_cycle_overloaded")

        # Lowercase nan and -inf are preserved as non-finite -> invalid metrics.
        for bad_val in ("nan", "-inf", "+Inf"):
            bad_text = (
                "freetpu_exporter_up 1\n"
                "freetpu_tpu_devices_discovered 1\n"
                'freetpu_tpu_health_status{chip="0"} 1\n'
                f'freetpu_tpu_duty_cycle_percent{{chip="0"}} {bad_val}\n'
            )
            telem_bad = summarize_tpu_samples(parse_prometheus_text(bad_text))
            ready_bad, reason_bad = evaluate_tpu_readiness(telem_bad)
            self.assertFalse(ready_bad)
            self.assertEqual(reason_bad, "tpu_exporter_invalid_metrics")

    def test_scalar_and_non_string_backend_models_rejected(self) -> None:
        for bad_models in ("my-model", 123, ["ok-model", 42], {"a": "b"}):
            with self.subTest(bad_models=bad_models):
                with self.assertRaises(ValueError):
                    BackendConfig.from_dict(
                        {
                            "id": "bad-models-b",
                            "base_url": "http://127.0.0.1:8000/v1",
                            "models": bad_models,
                        }
                    )
        with self.assertRaises(ValueError):
            BackendConfig(
                id="bad-models-direct",
                base_url="http://127.0.0.1:8000/v1",
                models="my-model",  # type: ignore[arg-type]
            )

        bad_cfg = Path(self._tmpdir.name) / "scalar_models.json"
        bad_cfg.write_text(
            json.dumps(
                {
                    "backends": [
                        {
                            "id": "scalar-m",
                            "base_url": "http://127.0.0.1:8000/v1",
                            "models": "my-model",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc = router_main(["--check-config", "--config", str(bad_cfg)])
        self.assertNotEqual(rc, 0)

    def test_partial_unhealthy_tpu_slice_marks_backend_unready(self) -> None:
        # 1. 7 of 8 chips healthy, 1 chip unhealthy -> unready ("tpu_unhealthy").
        partial_lines = [
            "freetpu_exporter_up 1",
            "freetpu_tpu_devices_discovered 8",
        ] + [
            f'freetpu_tpu_health_status{{chip="{i}"}} {0 if i == 3 else 1}'
            for i in range(8)
        ]
        telem_partial = summarize_tpu_samples(
            parse_prometheus_text("\n".join(partial_lines))
        )
        self.assertEqual(telem_partial.total_chips, 8)
        self.assertEqual(telem_partial.healthy_chips, 7)
        ready_p, reason_p = evaluate_tpu_readiness(telem_partial)
        self.assertFalse(ready_p)
        self.assertEqual(reason_p, "tpu_unhealthy")

        # 2. 8 discovered chips, only 7 health metrics emitted -> unready ("tpu_unhealthy").
        missing_lines = [
            "freetpu_exporter_up 1",
            "freetpu_tpu_devices_discovered 8",
        ] + [f'freetpu_tpu_health_status{{chip="{i}"}} 1' for i in range(7)]
        telem_missing = summarize_tpu_samples(
            parse_prometheus_text("\n".join(missing_lines))
        )
        self.assertEqual(telem_missing.total_chips, 8)
        self.assertEqual(telem_missing.healthy_chips, 7)
        ready_m, reason_m = evaluate_tpu_readiness(telem_missing)
        self.assertFalse(ready_m)
        self.assertEqual(reason_m, "tpu_unhealthy")

        # 3. All 8 of 8 chips healthy (with HBM telemetry) -> ready.
        healthy_lines = (
            [
                "freetpu_exporter_up 1",
                "freetpu_tpu_devices_discovered 8",
            ]
            + [f'freetpu_tpu_health_status{{chip="{i}"}} 1' for i in range(8)]
            + [f'freetpu_tpu_hbm_utilization_ratio{{chip="{i}"}} 0.25' for i in range(8)]
        )
        telem_ok = summarize_tpu_samples(
            parse_prometheus_text("\n".join(healthy_lines))
        )
        ready_ok, reason_ok = evaluate_tpu_readiness(telem_ok)
        self.assertTrue(ready_ok)
        self.assertIsNone(reason_ok)

    def test_upstream_logprobs_response_exceeding_request_json_node_limit_succeeds(
        self,
    ) -> None:
        # 1,024 generated tokens with 5 top_logprobs each produces >40,000 structural JSON nodes,
        # exceeding default inbound max_json_nodes (32,768) while fitting cleanly within
        # max_response_json_nodes and the shared response memory budget.
        token_entry = {
            "token": "ok",
            "logprob": -0.05,
            "top_logprobs": [
                {"token": f"t{k}", "logprob": -0.1 * (k + 1)} for k in range(5)
            ],
        }
        logprobs_payload = {
            "id": "chatcmpl-logprobs-big",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "m-logprobs",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok " * 1024},
                    "logprobs": {"content": [token_entry for _ in range(1024)]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1024, "total_tokens": 1036},
        }
        raw_nodes = _validate_json_structure(
            json.dumps(logprobs_payload).encode("utf-8"),
            max_depth=64,
            max_nodes=524288,
        )
        self.assertGreater(raw_nodes, 32768)

        backend = self._track(
            _MockOpenAIBackend(
                "logprobs-b",
                ["m-logprobs"],
                custom_non_stream_payload=logprobs_payload,
            )
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="logprobs-b",
                    base_url=backend.base_url,
                    models=["m-logprobs"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Valid 1,024-token top_logprobs=5 upstream response succeeds with HTTP 200.
        st, _, bd = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-logprobs",
                "messages": [{"role": "user", "content": "give logprobs"}],
            },
        )
        self.assertEqual(st, 200)
        parsed = json.loads(bd.decode("utf-8"))
        self.assertEqual(len(parsed["choices"][0]["logprobs"]["content"]), 1024)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

        # 2. Inbound request exceeding max_json_nodes is still rejected with HTTP 400.
        cfg_strict = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_json_nodes=500,
            max_response_json_nodes=1000,
            backends=[
                BackendConfig(
                    id="logprobs-b",
                    base_url=backend.base_url,
                    models=["m-logprobs"],
                )
            ],
        )
        router2 = self._track(create_router_server(cfg_strict))
        rport2 = router2.server_address[1]
        st_req_bad, _, bd_req_bad = _http_request(
            rport2,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-logprobs", "messages": [[] for _ in range(600)]},
        )
        self.assertEqual(st_req_bad, 400)
        self.assertIn(b"json_complexity_limit_exceeded", bd_req_bad)

        # 3. Upstream response exceeding explicit max_response_json_nodes (1000) returns HTTP 502.
        st_resp_bad, _, bd_resp_bad = _http_request(
            rport2,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-logprobs",
                "messages": [{"role": "user", "content": "hit resp cap"}],
            },
        )
        self.assertEqual(st_resp_bad, 502)
        self.assertIn(b"response_too_large", bd_resp_bad)
        self.assertEqual(router2.engine._resp_mem_used_bytes, 0)

    def test_url_parse_errors_redact_unicode_authority_credentials(self) -> None:
        # Unicode NFKC-unsafe character in authority makes CPython urlsplit() raise ValueError
        # whose own message quotes the raw authority; validate_http_url and scrape_tpu_exporter
        # must suppress that raw message and exception chain so credentials never leak.
        secret_urls = (
            "http://user:topsecret@bad\u2100host/v1",
            "http://user:topsecret@[::1:8000/v1",
        )
        for bad_url in secret_urls:
            with self.subTest(bad_url=bad_url):
                with self.assertRaises(ValueError) as ctx:
                    validate_http_url(bad_url, field_name="backend base_url", allow_query=False)
                self.assertNotIn("topsecret", str(ctx.exception))
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

                telem = scrape_tpu_exporter(bad_url, timeout_s=0.5)
                self.assertFalse(telem.reachable)
                self.assertNotIn("topsecret", telem.error or "")

    def test_ipv6_loopback_listener_binds_and_serves_requests(self) -> None:
        if not socket.has_ipv6:
            self.skipTest("IPv6 not supported in this Python build")
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind(("::1", 0))
        except OSError:
            self.skipTest("IPv6 loopback ::1 unavailable on this host")

        backend = self._track(_MockOpenAIBackend("ipv6-b", ["m-v6"]))
        cfg = RouterConfig(
            host="::1",
            port=0,
            backends=[
                BackendConfig(
                    id="ipv6-b",
                    base_url=backend.base_url,
                    models=["m-v6"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        self.assertEqual(router.address_family, socket.AF_INET6)
        rport = router.server_address[1]
        conn = http.client.HTTPConnection("::1", rport, timeout=3.0)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(body["status"], "ok")
        finally:
            conn.close()

    def test_backend_id_validates_safe_ascii_header_characters(self) -> None:
        for bad_id in ("bad\u2603id", "bad id", "bad\r\nid", "bad\tid", " leading"):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    BackendConfig(id=bad_id, base_url="http://127.0.0.1:8000/v1")

    def test_half_closed_client_shutdown_wr_receives_full_non_stream_and_sse_responses(
        self,
    ) -> None:
        primary = self._track(
            _MockOpenAIBackend("halfclose-primary", ["m-hc"], status_code=200)
        )
        backup = self._track(
            _MockOpenAIBackend("halfclose-backup", ["m-hc"], latency_s=0.08)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            backends=[
                BackendConfig(
                    id="halfclose-primary",
                    local=True,
                    base_url=primary.base_url,
                    models=["m-hc"],
                ),
                BackendConfig(
                    id="halfclose-backup",
                    local=False,
                    base_url=backup.base_url,
                    models=["m-hc"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        primary.status_code = 503

        # 1. Non-streaming POST with immediate shutdown(SHUT_WR) must still fail over
        # from 503 primary and receive the complete 200 JSON response from backup.
        req_body = json.dumps(
            {"model": "m-hc", "messages": [{"role": "user", "content": "half close"}]}
        ).encode("utf-8")
        sock = socket.create_connection(("127.0.0.1", rport), timeout=3.0)
        try:
            sock.sendall(
                (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{rport}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(req_body)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii")
                + req_body
            )
            sock.shutdown(socket.SHUT_WR)
            chunks: List[bytes] = []
            while True:
                part = sock.recv(4096)
                if not part:
                    break
                chunks.append(part)
            raw_resp = b"".join(chunks)
        finally:
            sock.close()
        self.assertIn(b"200 OK", raw_resp)
        self.assertIn(b"halfclose-backup", raw_resp)

        # 2. Streaming SSE POST with immediate shutdown(SHUT_WR) must receive all chunks + [DONE].
        sse_body = json.dumps(
            {
                "model": "m-hc",
                "stream": True,
                "messages": [{"role": "user", "content": "half close sse"}],
            }
        ).encode("utf-8")
        sock2 = socket.create_connection(("127.0.0.1", rport), timeout=3.0)
        try:
            sock2.sendall(
                (
                    f"POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{rport}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(sse_body)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode("ascii")
                + sse_body
            )
            sock2.shutdown(socket.SHUT_WR)
            chunks2: List[bytes] = []
            while True:
                part = sock2.recv(4096)
                if not part:
                    break
                chunks2.append(part)
            raw_sse = b"".join(chunks2)
        finally:
            sock2.close()
        self.assertIn(b"200 OK", raw_sse)
        self.assertIn(b"data: [DONE]", raw_sse)

    def test_bounded_http_response_accumulates_contiguously_and_budgets_final_copy(
        self,
    ) -> None:
        class _FragmentedChunkedResp:
            chunked = True
            status = 200
            length = None

            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._idx = 0

            def getheader(self, name: str) -> Optional[str]:
                return None

            def read1(self, amt: int) -> bytes:
                if self._idx >= len(self._payload):
                    return b""
                # Simulate 1-byte HTTP chunk frames.
                b = self._payload[self._idx : self._idx + 1]
                self._idx += 1
                return b

        # 1. Many 1-byte chunks accumulate cleanly and reserve 2x during bytes(buf),
        # then release the temporary copy reservation via release_fn.
        payload = b"x" * 1024
        used = [0]
        peak = [0]
        budget = 2048

        def _reserve(n: int) -> bool:
            if used[0] + n > budget:
                return False
            used[0] += n
            if used[0] > peak[0]:
                peak[0] = used[0]
            return True

        def _release(n: int) -> None:
            used[0] = max(0, used[0] - n)

        reserved_box = [0]
        out = _read_bounded_http_response(
            _FragmentedChunkedResp(payload),  # type: ignore[arg-type]
            None,
            max_bytes=4096,
            deadline=time.monotonic() + 5.0,
            reserve_fn=_reserve,
            release_fn=_release,
            reserved_out=reserved_box,
        )
        self.assertEqual(out, payload)
        self.assertEqual(peak[0], 2048)
        self.assertEqual(used[0], 1024)
        self.assertEqual(reserved_box[0], 1024)

        # 2. If budget covers accumulation (1024) but not the duplicate bytes(buf) copy
        # (2048), _read_bounded_http_response rejects before copying.
        used_tight = [0]
        reserved_tight = [0]
        budget_tight = 1500

        def _reserve_tight(n: int) -> bool:
            if used_tight[0] + n > budget_tight:
                return False
            used_tight[0] += n
            return True

        with self.assertRaises(_MemoryBudgetExceededError):
            _read_bounded_http_response(
                _FragmentedChunkedResp(payload),  # type: ignore[arg-type]
                None,
                max_bytes=4096,
                deadline=time.monotonic() + 5.0,
                reserve_fn=_reserve_tight,
                release_fn=lambda n: used_tight.__setitem__(0, max(0, used_tight[0] - n)),
                reserved_out=reserved_tight,
            )
        self.assertEqual(used_tight[0], 1024)
        self.assertEqual(reserved_tight[0], 1024)

    def test_permanent_request_memory_limit_returns_413_vs_contention_503(
        self,
    ) -> None:
        slow_b = self._track(
            _MockOpenAIBackend("mem-perm-b", ["m-perm"], latency_s=0.35)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_request_body_bytes=10000,
            max_total_request_memory_bytes=2000,
            backends=[
                BackendConfig(
                    id="mem-perm-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=slow_b.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Single ASCII request whose upfront 4x parse_budget exceeds total budget
        # (600 * 4 = 2400 > 2000) returns permanent 413 even when router is idle.
        st_upfront, _, bd_upfront = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-perm",
                "messages": [{"role": "user", "content": "x" * 540}],
            },
        )
        self.assertEqual(st_upfront, 413)
        self.assertIn(b"payload_too_large", bd_upfront)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

        # 2. Single request with \u escape whose post-read 9x estimate exceeds total budget
        # (280 * 9 = 2520 > 2000) returns permanent 413 even when router is idle.
        esc_body = (
            '{"model":"m-perm","messages":[{"role":"user","content":"'
            + ("x" * 210)
            + '\\u0041"}]}'
        ).encode("ascii")
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=esc_body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            bd_esc = resp.read()
            self.assertEqual(resp.status, 413)
            self.assertIn(b"payload_too_large", bd_esc)
        finally:
            conn.close()
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

        # 3. Request that DOES fit when idle (~450 * 4 = 1800 <= 2000) returns retryable 503
        # only under temporary concurrent contention, then succeeds when idle.
        fit_payload = {
            "model": "m-perm",
            "messages": [{"role": "user", "content": "y" * 380}],
        }
        first_res: Dict[str, Any] = {}

        def _send_first() -> None:
            st, _, _ = _http_request(
                rport, "POST", "/v1/chat/completions", body=fit_payload
            )
            first_res["status"] = st

        t1 = threading.Thread(target=_send_first, daemon=True)
        t1.start()
        deadline = time.monotonic() + 2.0
        while slow_b.current_inflight < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(slow_b.current_inflight, 1)

        st_busy, _, bd_busy = _http_request(
            rport, "POST", "/v1/chat/completions", body=fit_payload
        )
        self.assertEqual(st_busy, 503)
        self.assertIn(b"server_memory_limit_reached", bd_busy)

        t1.join(timeout=3.0)
        self.assertEqual(first_res.get("status"), 200)
        rel_deadline = time.monotonic() + 1.0
        while router.engine._req_mem_used_bytes != 0 and time.monotonic() < rel_deadline:
            time.sleep(0.01)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_prometheus_parser_preserves_samples_with_braces_in_quoted_labels(
        self,
    ) -> None:
        prom_text = "\n".join(
            [
                "# HELP freetpu_tpu_health_status TPU chip health",
                'freetpu_exporter_up{} 1',
                'freetpu_tpu_devices_discovered{meta="cluster}prod",} 2',
                'freetpu_tpu_health_status{chip="0",meta="a}b",note="q=\\"}end\\\\"} 1 1700000000',
                'freetpu_tpu_health_status{chip="1",literal_bs="\\\\n}",newline="l1\\nl2}"} 1',
                'freetpu_tpu_duty_cycle_percent{chip="0",desc="brace}inside"} 60.0',
                'freetpu_tpu_duty_cycle_percent{chip="1",desc="escaped \\"}\\""} 80.0',
                # Malformed unterminated quote must be ignored without breaking valid lines.
                'freetpu_tpu_health_status{chip="2",meta="unclosed} 0',
            ]
        )
        samples = parse_prometheus_text(prom_text)
        by_key = {(s.name, s.labels.get("chip", "")): s for s in samples}

        self.assertIn(("freetpu_exporter_up", ""), by_key)
        self.assertEqual(by_key[("freetpu_exporter_up", "")].value, 1.0)
        self.assertEqual(
            by_key[("freetpu_tpu_devices_discovered", "")].labels["meta"],
            "cluster}prod",
        )
        s0 = by_key[("freetpu_tpu_health_status", "0")]
        self.assertEqual(s0.labels["meta"], "a}b")
        self.assertEqual(s0.labels["note"], 'q="}end\\')
        s1 = by_key[("freetpu_tpu_health_status", "1")]
        self.assertEqual(s1.labels["literal_bs"], r"\n}")
        self.assertEqual(s1.labels["newline"], "l1\nl2}")
        self.assertNotIn(("freetpu_tpu_health_status", "2"), by_key)

        telem = summarize_tpu_samples(samples)
        self.assertTrue(telem.exporter_up)
        self.assertEqual(telem.devices_discovered, 2)
        self.assertEqual(telem.healthy_chips, 2)
        self.assertEqual(telem.total_chips, 2)
        self.assertAlmostEqual(telem.avg_duty_cycle_percent or 0.0, 70.0)
        self.assertAlmostEqual(telem.max_duty_cycle_percent or 0.0, 80.0)

    def test_jsonl_usage_write_failures_increment_counters_and_rate_limit_warnings(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("meter-err-b", ["m-meter"]))
        # Pointing JSONL path at an existing directory deterministically triggers
        # OSError (IsADirectoryError) even when tests run as root in CI containers.
        bad_jsonl_dir = Path(self._tmpdir.name) / "not_a_file_dir"
        bad_jsonl_dir.mkdir(exist_ok=True)
        err_stream = io.StringIO()
        usage_hook = JsonlUsageMeterHook(
            bad_jsonl_dir,
            error_log_interval_s=60.0,
            error_stream=err_stream,
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="meter-err-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=usage_hook))
        rport = router.server_address[1]

        for idx in range(3):
            st, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-meter",
                    "messages": [{"role": "user", "content": f"req-{idx}"}],
                },
            )
            self.assertEqual(st, 200)

        self.assertEqual(usage_hook.events_written, 0)
        self.assertEqual(usage_hook.write_errors, 3)
        self.assertEqual(router.engine.metrics.usage_meter_errors_total(), 3)

        st_m, _, mbody = _http_request(rport, "GET", "/metrics")
        self.assertEqual(st_m, 200)
        self.assertIn("freetpu_router_usage_meter_errors_total 3", mbody.decode("utf-8"))

        # Rate-limited: 3 rapid failures emit only 1 warning line.
        warn_lines = [ln for ln in err_stream.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(warn_lines), 1)
        self.assertIn("total_write_errors=1", warn_lines[0])

        # Simulate interval expiry and trigger a 4th failure -> emits 2nd warning with suppressed count.
        usage_hook._last_error_log_monotonic = time.monotonic() - 120.0
        st4, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-meter",
                "messages": [{"role": "user", "content": "req-3"}],
            },
        )
        self.assertEqual(st4, 200)
        self.assertEqual(usage_hook.write_errors, 4)
        self.assertEqual(router.engine.metrics.usage_meter_errors_total(), 4)
        warn_lines_after = [ln for ln in err_stream.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(warn_lines_after), 2)
        self.assertIn("total_write_errors=4", warn_lines_after[1])
        self.assertIn("2 suppressed", warn_lines_after[1])

    def test_bearer_auth_health_returns_minimal_unauthenticated_and_detailed_authenticated(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("secret-tpu-b", ["m-secret-model"]))
        exporter = self._track(_MockTpuExporter(health_status=1, generation="v6e"))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            auth={"type": "bearer", "tokens": ["health-secret-tok"]},
            backends=[
                BackendConfig(
                    id="secret-tpu-b",
                    type="cloud-tpu-v6e",
                    base_url=backend.base_url,
                    tpu_exporter_url=exporter.metrics_url,
                    metadata={"rack": "us-west-private-rack"},
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. Unauthenticated /health returns status-only payload with zero backend metadata.
        st_unauth, _, bd_unauth = _http_request(rport, "GET", "/health")
        self.assertEqual(st_unauth, 200)
        self.assertEqual(json.loads(bd_unauth), {"status": "ok"})
        text_unauth = bd_unauth.decode("utf-8")
        self.assertNotIn("secret-tpu-b", text_unauth)
        self.assertNotIn("m-secret-model", text_unauth)
        self.assertNotIn("us-west-private-rack", text_unauth)
        self.assertNotIn(backend.base_url, text_unauth)

        # 2. Invalid bearer token on exempt /health still returns only minimal status.
        st_bad, _, bd_bad = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Authorization": "Bearer wrong-token"},
        )
        self.assertEqual(st_bad, 200)
        self.assertEqual(json.loads(bd_bad), {"status": "ok"})

        # 3. Valid bearer token on /health returns full backend details, URL, models, and TPU telemetry.
        st_auth, _, bd_auth = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Authorization": "Bearer health-secret-tok"},
        )
        self.assertEqual(st_auth, 200)
        parsed_auth = json.loads(bd_auth)
        self.assertEqual(parsed_auth["status"], "ok")
        self.assertEqual(parsed_auth["healthy_backends"], 1)
        self.assertEqual(len(parsed_auth["backends"]), 1)
        snap = parsed_auth["backends"][0]
        self.assertEqual(snap["id"], "secret-tpu-b")
        self.assertEqual(snap["metadata"], {"rack": "us-west-private-rack"})
        self.assertIn("m-secret-model", snap["models"])
        self.assertIsNotNone(snap["tpu_telemetry"])

        # 4. Degraded state returns 503 minimal status unauthenticated and 503 full details authenticated.
        exporter.health_status = 0
        router.engine.refresh_all_backends()
        st_deg_unauth, _, bd_deg_unauth = _http_request(rport, "GET", "/health")
        self.assertEqual(st_deg_unauth, 503)
        self.assertEqual(json.loads(bd_deg_unauth), {"status": "degraded"})

        st_deg_auth, _, bd_deg_auth = _http_request(
            rport,
            "GET",
            "/health",
            headers={"Authorization": "Bearer health-secret-tok"},
        )
        self.assertEqual(st_deg_auth, 503)
        parsed_deg = json.loads(bd_deg_auth)
        self.assertEqual(parsed_deg["status"], "degraded")
        self.assertEqual(parsed_deg["healthy_backends"], 0)
        self.assertEqual(len(parsed_deg["backends"]), 1)

    def test_malformed_pre_commit_sse_json_fails_over_instead_of_committing_200(
        self,
    ) -> None:
        # Raw HTTP primary backend that passes /v1/models health probes but returns
        # malformed JSON (or non-object JSON) in its first SSE data frame.
        bad_payload_box = [b"data: {corrupted-json\n\n"]

        class _CorruptSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-corrupt-sse"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                body = bad_payload_box[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

        corrupt_srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _CorruptSSEHandler))
        corrupt_port = corrupt_srv.server_address[1]

        healthy_fallback = self._track(
            _MockOpenAIBackend("healthy-sse-fallback", ["m-corrupt-sse"])
        )

        # 1. With a healthy fallback, both malformed JSON and non-object JSON fail over cleanly.
        for bad_frame in (b"data: {corrupted-json\n\n", b"data: 12345\n\n"):
            with self.subTest(bad_frame=bad_frame):
                bad_payload_box[0] = bad_frame
                cfg = RouterConfig(
                    host="127.0.0.1",
                    port=0,
                    policy="local_first",
                    backends=[
                        BackendConfig(
                            id="corrupt-primary",
                            type="pixel-tpu",
                            local=True,
                            priority=0,
                            base_url=f"http://127.0.0.1:{corrupt_port}/v1",
                            models=["m-corrupt-sse"],
                        ),
                        BackendConfig(
                            id="healthy-sse-fallback",
                            type="cloud-tpu-v6e",
                            local=False,
                            priority=1,
                            base_url=healthy_fallback.base_url,
                            models=["m-corrupt-sse"],
                        ),
                    ],
                )
                router = self._track(create_router_server(cfg))
                rport = router.server_address[1]
                st, hdrs, bd = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-corrupt-sse",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(st, 200)
                self.assertEqual(hdrs["x-freetpu-backend"], "healthy-sse-fallback")
                self.assertEqual(hdrs["x-freetpu-failovers"], "1")
                self.assertIn(b"data: [DONE]", bd)
                self.assertNotIn(b"corrupted-json", bd)

        # 2. Without a fallback, malformed pre-commit JSON returns HTTP 502 instead of committing 200.
        bad_payload_box[0] = b"data: {corrupted-json\n\n"
        cfg_solo = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="corrupt-only",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{corrupt_port}/v1",
                    models=["m-corrupt-sse"],
                )
            ],
        )
        router_solo = self._track(create_router_server(cfg_solo))
        st_solo, _, bd_solo = _http_request(
            router_solo.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-corrupt-sse",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st_solo, 502)
        self.assertIn(b"upstream_stream_error", bd_solo)

    def test_json_memory_estimator_covers_dense_container_working_set(self) -> None:
        def _deep_sizeof(obj: Any, seen: Optional[set[int]] = None) -> int:
            if seen is None:
                seen = set()
            oid = id(obj)
            if oid in seen:
                return 0
            seen.add(oid)
            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    total += _deep_sizeof(k, seen) + _deep_sizeof(v, seen)
            elif isinstance(obj, list):
                for item in obj:
                    total += _deep_sizeof(item, seen)
            return total

        dense_cases = [
            [{"a": 1} for _ in range(10000)],
            [{} for _ in range(10000)],
            [{f"k{i}": f"v{i}"} for i in range(5000)],
        ]
        for payload in dense_cases:
            raw = json.dumps(payload).encode("utf-8")
            containers = [0]
            nodes = _validate_json_structure(
                raw,
                max_depth=64,
                max_nodes=262144,
                containers_out=containers,
            )
            est = _estimate_json_parse_memory(
                raw, nodes=nodes, containers=containers[0]
            )
            text = raw.decode("utf-8")
            parsed = json.loads(text)
            actual = sys.getsizeof(raw) + sys.getsizeof(text) + _deep_sizeof(parsed)
            self.assertGreaterEqual(
                est,
                actual,
                f"Estimated {est} bytes < actual {actual} bytes for dense payload",
            )

        # End-to-end: 10,000 one-field objects (~100 KB wire, ~2.1 MB actual) is rejected
        # with 413 against a 1.5 MB total request memory budget instead of slipping through.
        backend = self._track(_MockOpenAIBackend("dense-mem-b", ["m-dense"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_json_nodes=65536,
            max_total_request_memory_bytes=1_500_000,
            backends=[
                BackendConfig(
                    id="dense-mem-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        st, _, bd = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-dense",
                "messages": [{"role": "user", "content": "dense"}],
                "extra_items": [{"a": 1} for _ in range(10000)],
            },
        )
        self.assertEqual(st, 413)
        self.assertIn(b"payload_too_large", bd)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_prometheus_parser_splits_only_on_lf_preserving_cr_and_unicode_line_separators(
        self,
    ) -> None:
        prom_text = (
            'freetpu_exporter_up 1\r\n'
            'freetpu_tpu_devices_discovered 3\n'
            'freetpu_tpu_health_status{chip="0",desc="before\rafter"} 1\n'
            'freetpu_tpu_health_status{chip="1",desc="ls\u2028here"} 1\n'
            'freetpu_tpu_health_status{chip="2",desc="ps\u2029here"} 1\r\n'
            'freetpu_tpu_duty_cycle_percent{chip="0",desc="cr\rlabel"} 30.0\n'
            'freetpu_tpu_duty_cycle_percent{chip="1",desc="u2028\u2028label"} 60.0\n'
            'freetpu_tpu_duty_cycle_percent{chip="2",desc="u2029\u2029label"} 90.0\n'
        )
        samples = parse_prometheus_text(prom_text)
        by_key = {(s.name, s.labels.get("chip", "")): s for s in samples}
        self.assertEqual(
            by_key[("freetpu_tpu_health_status", "0")].labels["desc"],
            "before\rafter",
        )
        self.assertEqual(
            by_key[("freetpu_tpu_health_status", "1")].labels["desc"],
            "ls\u2028here",
        )
        self.assertEqual(
            by_key[("freetpu_tpu_health_status", "2")].labels["desc"],
            "ps\u2029here",
        )
        telem = summarize_tpu_samples(samples)
        self.assertTrue(telem.exporter_up)
        self.assertEqual(telem.devices_discovered, 3)
        self.assertEqual(telem.healthy_chips, 3)
        self.assertEqual(telem.total_chips, 3)
        self.assertAlmostEqual(telem.avg_duty_cycle_percent or 0.0, 60.0)

    def test_stalled_upload_does_not_hold_parse_budget_or_block_concurrent_requests(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("stall-up-b", ["m-stallup"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_request_body_bytes=10000,
            max_total_request_memory_bytes=2000,
            client_read_timeout_s=0.6,
            backends=[
                BackendConfig(
                    id="stall-up-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Open two stalled uploads each declaring Content-Length: 480 (which would
        # have reserved 1,920 bytes upfront under 4x pre-reservation) and sending only 1 byte.
        stalled_socks: List[socket.socket] = []
        try:
            for _ in range(2):
                s = socket.create_connection(("127.0.0.1", rport), timeout=2.0)
                stalled_socks.append(s)
                s.sendall(
                    (
                        f"POST /v1/chat/completions HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{rport}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: 480\r\n"
                        f"Connection: close\r\n\r\n"
                        "{"
                    ).encode("ascii")
                )
            deadline = time.monotonic() + 1.5
            while router.engine._req_mem_used_bytes < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            # Only the 2 actually received bytes are charged — NOT 2 * 1,920 = 3,840 bytes.
            self.assertEqual(router.engine._req_mem_used_bytes, 2)

            # Concurrent real inference request (~380 bytes -> ~1,520 parse peak) succeeds with 200!
            st_ok, _, bd_ok = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-stallup",
                    "messages": [{"role": "user", "content": "a" * 310}],
                },
            )
            self.assertEqual(st_ok, 200)
            self.assertIn(b"stall-up-b", bd_ok)
        finally:
            for s in stalled_socks:
                s.close()

        # Once stalled sockets close, all incrementally reserved bytes are released cleanly.
        deadline = time.monotonic() + 2.0
        while router.engine._req_mem_used_bytes != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_streamed_delta_audio_commits_immediately_and_advances_progress(
        self,
    ) -> None:
        release_rest = threading.Event()
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _AudioSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-audio"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def _emit(payload: Dict[str, Any]) -> None:
                    self.wfile.write(
                        f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()

                # Role-only preamble followed immediately by first audio-only chunk.
                _emit(
                    {
                        "id": "chatcmpl-audio",
                        "object": "chat.completion.chunk",
                        "model": "m-audio",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                _emit(
                    {
                        "id": "chatcmpl-audio",
                        "object": "chat.completion.chunk",
                        "model": "m-audio",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"audio": {"id": "aud_1", "data": "A" * 512}},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                # Wait until client confirms it received the first audio chunk live.
                release_rest.wait(timeout=3.0)
                # Stream 10 more audio-only chunks (>5 KiB total, exceeding 4 KiB pre-commit cap,
                # and spanning >550ms > 450ms request_timeout_s to prove audio advances progress).
                for idx in range(10):
                    time.sleep(0.055)
                    _emit(
                        {
                            "id": "chatcmpl-audio",
                            "object": "chat.completion.chunk",
                            "model": "m-audio",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"audio": {"data": f"{idx:02d}" + ("B" * 510)}},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                _emit(
                    {
                        "id": "chatcmpl-audio",
                        "object": "chat.completion.chunk",
                        "model": "m-audio",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"audio": {"transcript": "spoken audio hello"}},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _AudioSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_sse_pre_commit_bytes=4096,
            request_timeout_s=0.45,
            backends=[
                BackendConfig(
                    id="audio-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-audio"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        rport = router.server_address[1]

        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=5.0)
        try:
            body = json.dumps(
                {
                    "model": "m-audio",
                    "stream": True,
                    "messages": [{"role": "user", "content": "speak"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            received = bytearray()
            # Read until the first audio chunk arrives live BEFORE releasing the rest.
            while b'"aud_1"' not in received:
                part = resp.read1(4096)
                if not part:
                    break
                received.extend(part)
            self.assertIn(b'"aud_1"', received)
            release_rest.set()
            received.extend(resp.read())
        finally:
            release_rest.set()
            conn.close()

        self.assertIn(b"spoken audio hello", received)
        self.assertIn(b"data: [DONE]", received)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertGreater(recorded_usage[0].completion_tokens, 0)
        self.assertEqual(router.engine.metrics.total_inflight(), 0)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

    def test_pre_commit_short_keepalive_lines_buffered_contiguously_within_budget(
        self,
    ) -> None:
        keepalive_count_box = [1000]

        class _KeepaliveSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-ka-contig"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                body = (
                    (b":\n\n" * keepalive_count_box[0])
                    + b'data: {"id":"chatcmpl-ka","object":"chat.completion.chunk","model":"m-ka-contig","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                    + b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                except OSError:
                    pass

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _KeepaliveSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_sse_pre_commit_bytes=8192,
            max_total_response_memory_bytes=4096,
            backends=[
                BackendConfig(
                    id="ka-contig-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-ka-contig"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # 1. 1,000 short keepalive lines (3,000 B + ~180 B output <= 4,096 B budget) succeed cleanly.
        keepalive_count_box[0] = 1000
        st_ok, _, bd_ok = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ka-contig",
                "stream": True,
                "messages": [{"role": "user", "content": "ka ok"}],
            },
        )
        self.assertEqual(st_ok, 200)
        self.assertIn(b"data: [DONE]", bd_ok)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

        # 2. 1,500 short keepalive lines (4,500 B > 4,096 B budget) rejected cleanly with 503.
        keepalive_count_box[0] = 1500
        st_over, _, bd_over = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ka-contig",
                "stream": True,
                "messages": [{"role": "user", "content": "ka over"}],
            },
        )
        self.assertEqual(st_over, 503)
        self.assertIn(b"server_memory_limit_reached", bd_over)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)
        self.assertTrue(router.engine._backends[0].healthy)

    def test_stream_include_usage_missing_final_usage_and_done_surfaces_truncation(
        self,
    ) -> None:
        include_usage_chunk_box = [False]
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _CloseDelimSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-incusage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-incusage",'
                    b'"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
                    b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-incusage",'
                    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                )
                if include_usage_chunk_box[0]:
                    self.wfile.write(
                        b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-incusage",'
                        b'"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
                    )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _CloseDelimSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="incusage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-incusage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        rport = router.server_address[1]

        # 1. Missing both [DONE] and requested usage chunk -> surfaced as truncation (502).
        include_usage_chunk_box[0] = False
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-incusage",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)

        # 2. When the requested final usage chunk arrives before close-delimited EOF -> clean 200.
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        include_usage_chunk_box[0] = True
        st_ok, _, bd_ok = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-incusage",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi again"}],
            },
        )
        self.assertEqual(st_ok, 200)
        self.assertIn(b'"completion_tokens":3', bd_ok)
        self.assertEqual(len(recorded_usage), 2)
        self.assertEqual(recorded_usage[1].status_code, 200)
        self.assertEqual(recorded_usage[1].prompt_tokens, 7)
        self.assertEqual(recorded_usage[1].completion_tokens, 3)

    def test_sse_bare_cr_and_split_crlf_delimiters_handled_pre_and_post_commit(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _CRSseHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-crsse"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Pre-commit bare CR events (role-only + first token).
                self.wfile.write(
                    b'data: {"id":"cr1","object":"chat.completion.chunk","model":"m-crsse",'
                    b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\r\r'
                    b'data: {"id":"cr1","object":"chat.completion.chunk","model":"m-crsse",'
                    b'"choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}\r\r'
                )
                self.wfile.flush()
                time.sleep(0.03)
                # Post-commit chunk-split CRLF (\r at end of chunk 1, \n at start of chunk 2)
                # followed by bare CR terminator event.
                self.wfile.write(
                    b'data: {"id":"cr1","object":"chat.completion.chunk","model":"m-crsse",'
                    b'"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\r'
                )
                self.wfile.flush()
                time.sleep(0.03)
                self.wfile.write(b"\n\r")
                self.wfile.flush()
                time.sleep(0.03)
                self.wfile.write(b"\ndata: [DONE]\r\r")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _CRSseHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="crsse-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-crsse"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        st, _, bd = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-crsse",
                "stream": True,
                "messages": [{"role": "user", "content": "cr test"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(b"Hello ", bd)
        self.assertIn(b"world", bd)
        self.assertIn(b"data: [DONE]", bd)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertGreater(recorded_usage[0].completion_tokens, 0)
        self.assertTrue(router.engine._backends[0].healthy)

    def test_hook_role_validation_rejects_mismatched_hooks_at_check_config_and_startup(
        self,
    ) -> None:
        base_backend = BackendConfig(
            id="hook-val-b",
            type="pixel-tpu",
            base_url="http://127.0.0.1:8081/v1",
        )
        with self.assertRaisesRegex(ValueError, "authenticate"):
            build_hooks_from_config(
                RouterConfig(
                    backends=[base_backend],
                    auth={"hook": "router.hooks:NoOpScalerHook"},
                )
            )
        with self.assertRaisesRegex(ValueError, "record_usage"):
            build_hooks_from_config(
                RouterConfig(
                    backends=[base_backend],
                    usage_meter={"hook": "router.hooks:AllowAllAuthHook"},
                )
            )
        with self.assertRaisesRegex(ValueError, "on_queue_pressure"):
            build_hooks_from_config(
                RouterConfig(
                    backends=[base_backend],
                    scaler={"hook": "router.hooks:NoOpUsageMeterHook"},
                )
            )

        bad_cfg_path = Path(self._tmpdir.name) / "bad_hook_role.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "auth": {"hook": "router.hooks:NoOpUsageMeterHook"},
                    "backends": [
                        {
                            "id": "hook-val-b",
                            "type": "pixel-tpu",
                            "base_url": "http://127.0.0.1:8081/v1",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc = router_main(["--config", str(bad_cfg_path), "--check-config"])
        self.assertEqual(rc, 1)
        self.assertIn("authenticate", err_buf.getvalue())

    def test_tpu_exporter_stream_filters_unrelated_metrics_and_bounds_memory(
        self,
    ) -> None:
        # 1. Stream ~1.2 MiB of 300,000 unrelated short lines + oversized unrelated line
        # + valid TPU lines in small chunks (so the test server itself does not allocate
        # 1.2 MiB inside the same process's tracemalloc window) and verify scraper peak
        # heap stays well under 256 KiB.
        unrelated_chunk = b"a 1\n" * 2000  # 8 KiB per write x 150 = 1.2 MB, 300k lines
        oversized_unrelated = (f'other_metric{{blob="{"x" * 20000}"}} 1\n').encode("ascii")
        valid_tpu = (
            'freetpu_exporter_up 1\n'
            'freetpu_tpu_devices_discovered 2\n'
            'freetpu_tpu_health_status{chip="0",generation="v6e",extra="' + ("z" * 1024) + '"} 1\n'
            'freetpu_tpu_health_status{chip="1",generation="v6e"} 1\n'
            'freetpu_tpu_duty_cycle_percent{chip="0"} 40.0\n'
            'freetpu_tpu_duty_cycle_percent{chip="1"} 60.0\n'
        ).encode("utf-8")
        total_len = (len(unrelated_chunk) * 150) + len(oversized_unrelated) + len(valid_tpu)

        class _StreamingExporterHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(total_len))
                self.send_header("Connection", "close")
                self.end_headers()
                for _ in range(150):
                    self.wfile.write(unrelated_chunk)
                self.wfile.write(oversized_unrelated)
                self.wfile.write(valid_tpu)
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _StreamingExporterHandler))
        metrics_url = f"http://127.0.0.1:{srv.server_address[1]}/metrics"

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            telem = scrape_tpu_exporter(metrics_url, timeout_s=5.0)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        self.assertTrue(telem.reachable)
        self.assertTrue(telem.exporter_up)
        self.assertEqual(telem.devices_discovered, 2)
        self.assertEqual(telem.healthy_chips, 2)
        self.assertAlmostEqual(telem.avg_duty_cycle_percent or 0.0, 50.0)
        self.assertEqual(telem.generations, ["v6e"])
        # Streaming filter avoids retaining the 1.2 MB wire body or 300k PromSample/dict objects.
        self.assertLess(peak_bytes, 256 * 1024)

        # 2. Hostile exporter emitting >256 distinct TPU chip IDs fails closed cleanly.
        many_chips = "freetpu_exporter_up 1\n" + "".join(
            f'freetpu_tpu_health_status{{chip="c{i}"}} 1\n' for i in range(300)
        )
        exporter_many = self._track(_MockTpuExporter(raw_body=many_chips))
        telem_many = scrape_tpu_exporter(exporter_many.metrics_url, timeout_s=3.0)
        self.assertTrue(telem_many.reachable)
        self.assertFalse(telem_many.exporter_up)
        self.assertEqual(telem_many.error, "exporter_chip_limit_exceeded")

    def test_decoded_stream_options_released_before_inference(self) -> None:
        slow_b = self._track(
            _MockOpenAIBackend("rel-opts-b", ["m-relopts"], latency_s=0.35)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="rel-opts-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=slow_b.base_url,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        marker = "unique-stream-opts-sentinel-987654"
        req_body = {
            "model": "m-relopts",
            "stream": False,
            "stream_options": {"include_usage": True, "blob": [marker] * 50},
            "messages": [{"role": "user", "content": "hi"}],
        }
        res: Dict[str, Any] = {}

        def _run() -> None:
            st, _, _ = _http_request(
                rport, "POST", "/v1/chat/completions", body=req_body
            )
            res["status"] = st

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        deadline = time.monotonic() + 2.0
        while slow_b.current_inflight < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(slow_b.current_inflight, 1)

        # Inspect active do_POST frame while request is blocked in _handle_inference:
        # no decoded dict / stream_options container may remain in do_POST locals.
        saw_do_post = False
        for frame in sys._current_frames().values():
            cur: Any = frame
            while cur is not None:
                if cur.f_code.co_name == "do_POST":
                    saw_do_post = True
                    for var_name, val in cur.f_locals.items():
                        self.assertNotIn(
                            var_name,
                            ("req_payload", "raw_stream_opts", "req_text"),
                        )
                        self.assertFalse(
                            isinstance(val, dict) and "include_usage" in val,
                            f"do_POST retained decoded stream_options in local {var_name!r}",
                        )
                cur = cur.f_back
        self.assertTrue(saw_do_post)

        t.join(timeout=3.0)
        self.assertEqual(res.get("status"), 200)
        rel_deadline = time.monotonic() + 1.0
        while router.engine._req_mem_used_bytes != 0 and time.monotonic() < rel_deadline:
            time.sleep(0.01)
        self.assertEqual(router.engine._req_mem_used_bytes, 0)

    def test_tpu_readiness_counts_all_observed_chips_when_discovery_missing_or_undercounts(
        self,
    ) -> None:
        # 1. Missing devices_discovered: chip 0 healthy, chip 1 reports HBM/duty only -> NOT ready.
        missing_disc = "\n".join(
            [
                "freetpu_exporter_up 1",
                'freetpu_tpu_health_status{chip="0"} 1',
                'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25',
                'freetpu_tpu_hbm_utilization_ratio{chip="1"} 0.40',
                'freetpu_tpu_duty_cycle_percent{chip="1"} 20.0',
            ]
        )
        telem_missing = summarize_tpu_samples(parse_prometheus_text(missing_disc))
        self.assertEqual(telem_missing.total_chips, 2)
        self.assertEqual(telem_missing.healthy_chips, 1)
        ready, reason = evaluate_tpu_readiness(telem_missing)
        self.assertFalse(ready)
        self.assertEqual(reason, "tpu_unhealthy")

        # 2. Undercounted devices_discovered=1 with 2 observed chips -> still NOT ready.
        undercount_disc = "freetpu_tpu_devices_discovered 1\n" + missing_disc
        telem_under = summarize_tpu_samples(parse_prometheus_text(undercount_disc))
        self.assertEqual(telem_under.total_chips, 2)
        self.assertEqual(telem_under.healthy_chips, 1)
        ready2, reason2 = evaluate_tpu_readiness(telem_under)
        self.assertFalse(ready2)
        self.assertEqual(reason2, "tpu_unhealthy")

    def test_prometheus_parser_honors_zero_and_negative_max_samples(self) -> None:
        text = "freetpu_exporter_up 1\nfreetpu_tpu_devices_discovered 4\n"
        self.assertEqual(parse_prometheus_text(text, max_samples=0), [])
        self.assertEqual(parse_prometheus_text(text, max_samples=-3), [])
        one = parse_prometheus_text(text, max_samples=1)
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0].name, "freetpu_exporter_up")

    def test_stream_include_usage_interim_usage_without_final_usage_surfaces_truncation(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _InterimUsageSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-interim"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # vLLM continuous-usage style: non-empty choices chunk carries interim usage,
                # followed by EOF BEFORE final empty-choices usage frame and [DONE].
                self.wfile.write(
                    b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-interim",'
                    b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],'
                    b'"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n'
                    b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-interim",'
                    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                    b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
                )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _InterimUsageSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="interim-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-interim"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        conn = http.client.HTTPConnection(
            "127.0.0.1", router.server_address[1], timeout=3.0
        )
        try:
            body = json.dumps(
                {
                    "model": "m-interim",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)

    def test_tpu_exporter_multi_address_connect_falls_back_when_first_addr_stalls(
        self,
    ) -> None:
        from router import tpu_telemetry as telemetry_mod

        exporter = self._track(_MockTpuExporter(health_status=1, generation="v6e"))
        exp_port = exporter.server_address[1]
        stalled_addr = ("198.51.100.1", exp_port)
        good_addr = ("127.0.0.1", exp_port)
        fake_infos = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", stalled_addr),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", good_addr),
        ]
        orig_connect = socket.socket.connect
        seen_timeouts: List[Optional[float]] = []

        def _fake_connect(sock_self: socket.socket, address: Any) -> Any:
            if address == stalled_addr:
                t = sock_self.gettimeout()
                seen_timeouts.append(t)
                if t is not None and t > 0:
                    time.sleep(min(t, 0.35))
                raise TimeoutError("timed out")
            return orig_connect(sock_self, address)

        t0 = time.monotonic()
        with (
            mock.patch.object(
                telemetry_mod, "_resolve_host_with_deadline", return_value=fake_infos
            ),
            mock.patch.object(socket.socket, "connect", new=_fake_connect),
        ):
            telem = scrape_tpu_exporter(
                f"http://dual-stack-tpu.local:{exp_port}/metrics",
                timeout_s=0.6,
            )
        elapsed = time.monotonic() - t0

        self.assertTrue(telem.reachable)
        self.assertTrue(telem.exporter_up)
        self.assertEqual(telem.healthy_chips, 1)
        self.assertEqual(len(seen_timeouts), 1)
        self.assertIsNotNone(seen_timeouts[0])
        self.assertLess(seen_timeouts[0] or 99.0, 0.4)
        self.assertLess(elapsed, 0.6)

    def test_prompt_token_estimate_includes_tool_schemas_and_tool_call_history(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        backend = self._track(
            _MockOpenAIBackend(
                "tool-prompt-b",
                ["m-toolprompt"],
                custom_sse_chunks=[
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": "m-toolprompt",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ],
            )
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="tool-prompt-b",
                    type="cloud-tpu-v6e",
                    base_url=backend.base_url,
                    cost_per_1m_prompt_usd=1.0,
                    cost_per_1m_completion_usd=2.0,
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        tools_catalog = [
            {
                "type": "function",
                "function": {
                    "name": f"search_codebase_{i}",
                    "description": "Search repository files with regex and glob filters " * 4,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            }
            for i in range(6)
        ]
        prior_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_codebase_0",
                    "arguments": json.dumps({"pattern": "estimate_prompt_tokens", "path": "router/" + ("a" * 200)}),
                },
            }
        ]
        st, _, _ = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-toolprompt",
                "stream": True,
                "tools": tools_catalog,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": None, "tool_calls": prior_tool_calls},
                    {"role": "tool", "name": "search_codebase_0", "tool_call_id": "call_1", "content": "ok"},
                ],
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(len(recorded_usage), 1)
        # Without tools/tool_calls, "hi"+"ok" would estimate ~1 token; with them >400 tokens.
        self.assertGreater(recorded_usage[0].prompt_tokens, 400)
        self.assertGreater(recorded_usage[0].estimated_cost_usd, 0.0004)

    def test_stream_include_usage_withholds_done_when_final_usage_missing(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []
        pre_commit_only_done = [False]

        class _CaptureHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _DoneNoUsageHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-donenousage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if not pre_commit_only_done[0]:
                    self.wfile.write(
                        b'data: {"id":"c1","object":"chat.completion.chunk","model":"m-donenousage",'
                        b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
                    )
                    self.wfile.flush()
                    time.sleep(0.02)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _DoneNoUsageHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="donenousage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-donenousage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_CaptureHook()))
        rport = router.server_address[1]

        # 1. Post-commit [DONE] without final usage under include_usage=true -> [DONE] withheld, 502 recorded.
        pre_commit_only_done[0] = False
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-donenousage",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)

        # 2. Pre-commit [DONE] without usage under include_usage=true -> rejected 502 before committing 200.
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        pre_commit_only_done[0] = True
        st_pre, _, bd_pre = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-donenousage",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st_pre, 502)
        self.assertIn(b"upstream_stream_error", bd_pre)

    def test_streamed_choice_token_ids_and_logprobs_commit_while_prompt_token_ids_buffers(
        self,
    ) -> None:
        # Role-only + top-level prompt_token_ids / empty logprobs must keep buffering.
        self.assertFalse(
            _sse_event_should_commit(
                json.dumps(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "prompt_token_ids": [1, 2, 3],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "logprobs": {"content": []},
                                "finish_reason": None,
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        )
        # Choice-level token_ids commits immediately.
        self.assertTrue(
            _sse_event_should_commit(
                json.dumps(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "token_ids": [42],
                                "finish_reason": None,
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        )
        # Choice-level non-empty logprobs commits immediately.
        self.assertTrue(
            _sse_event_should_commit(
                json.dumps(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": {
                                    "content": [{"token": "x", "logprob": -0.1}]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        )

        # End-to-end live commit proof for a token_ids-only chunk before upstream releases rest.
        release_rest = threading.Event()

        class _TokenMetaSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-tokmeta"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"tm1","object":"chat.completion.chunk","prompt_token_ids":[1,2],'
                    b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                    b'data: {"id":"tm1","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{},"token_ids":[42],"finish_reason":null}]}\n\n'
                )
                self.wfile.flush()
                release_rest.wait(timeout=3.0)
                self.wfile.write(
                    b'data: {"id":"tm1","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{},"logprobs":{"content":[{"token":"ok","logprob":-0.1}]},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _TokenMetaSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="tokmeta-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-tokmeta"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        conn = http.client.HTTPConnection(
            "127.0.0.1", router.server_address[1], timeout=3.0
        )
        try:
            body = json.dumps(
                {
                    "model": "m-tokmeta",
                    "stream": True,
                    "messages": [{"role": "user", "content": "meta"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            seen = bytearray()
            while b'"token_ids":[42]' not in seen:
                part = resp.read1(4096)
                if not part:
                    break
                seen.extend(part)
            self.assertIn(b'"token_ids":[42]', seen)
            release_rest.set()
            seen.extend(resp.read())
        finally:
            release_rest.set()
            conn.close()
        self.assertIn(b"data: [DONE]", seen)

    def test_malformed_committed_sse_event_marks_failure_and_withholds_done(self) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        payload_mode = ["malformed_then_done"]

        class _MalformedPostCommitSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-malformed-sse"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if payload_mode[0] == "malformed_then_done":
                    self.wfile.write(
                        b'data: {"id":"m1","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                    )
                    self.wfile.flush()
                    self.wfile.write(b"data: {corrupted_json\n\ndata: [DONE]\n\n")
                else:
                    self.wfile.write(
                        b'data: {"id":"m2","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{"content":"Done"},"finish_reason":"stop"}]}\n\n'
                    )
                    self.wfile.flush()
                    self.wfile.write(b"data: [1, 2, 3]\n\n")
                self.wfile.flush()

        srv = self._track(
            ThreadingHTTPServer(("127.0.0.1", 0), _MalformedPostCommitSSEHandler)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="malformed-sse-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-malformed-sse"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Malformed JSON followed by [DONE]: withhold [DONE], emit error, record 502, mark unhealthy.
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-malformed-sse",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)
        self.assertEqual(router.engine._backends[0].total_failures, 1)
        self.assertEqual(
            router.engine._backends[0].unhealthy_reason, "stream_truncated_mid_response"
        )
        self.assertGreater(
            router.engine._backends[0].cooldown_until_monotonic, time.monotonic()
        )

        # 2. Non-object JSON after finish_reason="stop": still record 502 and mark failed/cooldown.
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        payload_mode[0] = "non_object_after_stop"
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-malformed-sse",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi2"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 2)
        self.assertEqual(recorded_usage[1].status_code, 502)
        self.assertEqual(router.engine._backends[0].total_failures, 2)
        self.assertGreater(
            router.engine._backends[0].cooldown_until_monotonic, time.monotonic()
        )

    def test_cors_allowlist_matches_explicit_default_ports_and_case(self) -> None:
        backend = self._track(_MockOpenAIBackend("cors-port-b", ["m-corsport"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            cors_allowed_origins=[
                "https://App.Example:443",
                "http://plain.example:80",
                "https://[::1]:443",
                "https://custom.example:8443",
            ],
            backends=[
                BackendConfig(
                    id="cors-port-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                    models=["m-corsport"],
                )
            ],
        )
        cfg.validate()
        self.assertEqual(
            cfg.cors_allowed_origins,
            [
                "https://app.example",
                "http://plain.example",
                "https://[::1]",
                "https://custom.example:8443",
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        # Preflight and POST with browser-serialized default-port origins succeed.
        st_opt, hdr_opt, _ = _http_request(
            rport,
            "OPTIONS",
            "/v1/chat/completions",
            headers={
                "Origin": "https://app.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(st_opt, 204)
        self.assertEqual(
            hdr_opt.get("access-control-allow-origin"), "https://app.example"
        )

        for ok_origin in (
            "https://app.example",
            "https://app.example:443",
            "http://plain.example",
            "https://[::1]",
            "https://custom.example:8443",
        ):
            st_ok, hdr_ok, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                headers={"Origin": ok_origin},
                body={
                    "model": "m-corsport",
                    "messages": [{"role": "user", "content": "ok"}],
                },
            )
            self.assertEqual(st_ok, 200, f"expected 200 for Origin={ok_origin}")
            self.assertEqual(hdr_ok.get("access-control-allow-origin"), ok_origin)

        # Mismatched scheme or non-default port must still be rejected with 403.
        for bad_origin in (
            "http://app.example",
            "https://app.example:8443",
            "https://custom.example",
        ):
            st_bad, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                headers={"Origin": bad_origin},
                body={
                    "model": "m-corsport",
                    "messages": [{"role": "user", "content": "no"}],
                },
            )
            self.assertEqual(st_bad, 403, f"expected 403 for Origin={bad_origin}")

    def test_terminal_choice_usage_accepted_for_completions_and_chat_while_continuous_usage_enforced(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        stream_mode = ["legacy_terminal_usage"]

        class _LegacyUsageSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-legacy-usage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if stream_mode[0] == "legacy_terminal_usage":
                    self.wfile.write(
                        b'data: {"id":"cmpl-1","object":"text_completion",'
                        b'"choices":[{"index":0,"text":"Hello","finish_reason":null}]}\n\n'
                        b'data: {"id":"cmpl-1","object":"text_completion",'
                        b'"choices":[{"index":0,"text":" world","finish_reason":"stop"}],'
                        b'"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}\n\n'
                        b"data: [DONE]\n\n"
                    )
                elif stream_mode[0] == "chat_llama_terminal_usage":
                    # llama-server b6210 style: no interim usage on non-terminal chunks,
                    # final usage attached to the terminal chat choice chunk.
                    self.wfile.write(
                        b'data: {"id":"chat-llama-1","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                        b'data: {"id":"chat-llama-1","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                        b'"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
                        b"data: [DONE]\n\n"
                    )
                elif stream_mode[0] == "legacy_interim_only":
                    self.wfile.write(
                        b'data: {"id":"cmpl-2","object":"text_completion",'
                        b'"choices":[{"index":0,"text":"Hello","finish_reason":null}],'
                        b'"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}\n\n'
                        b'data: {"id":"cmpl-2","object":"text_completion",'
                        b'"choices":[{"index":0,"text":" world","finish_reason":"stop"}]}\n\n'
                        b"data: [DONE]\n\n"
                    )
                else:
                    # vLLM continuous_usage_stats=true: interim usage on non-terminal chunk
                    # AND terminal choice chunk, but missing final choices:[] usage frame.
                    self.wfile.write(
                        b'data: {"id":"chat-vllm-1","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],'
                        b'"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}\n\n'
                        b'data: {"id":"chat-vllm-1","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                        b'"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
                        b"data: [DONE]\n\n"
                    )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _LegacyUsageSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="legacy-usage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-legacy-usage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. /v1/completions with llama-server terminal choice + usage succeeds cleanly.
        st_ok, _, bd_ok = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-legacy-usage",
                "prompt": "Say hello",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(st_ok, 200)
        self.assertIn(b"data: [DONE]", bd_ok)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 12)
        self.assertEqual(recorded_usage[0].completion_tokens, 4)
        self.assertEqual(router.engine._backends[0].total_failures, 0)

        # 2. /v1/chat/completions with llama-server b6210 terminal choice + usage succeeds cleanly.
        stream_mode[0] = "chat_llama_terminal_usage"
        st_chat, _, bd_chat = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-legacy-usage",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st_chat, 200)
        self.assertIn(b"data: [DONE]", bd_chat)
        self.assertEqual(len(recorded_usage), 2)
        self.assertEqual(recorded_usage[1].status_code, 200)
        self.assertEqual(recorded_usage[1].prompt_tokens, 10)
        self.assertEqual(recorded_usage[1].completion_tokens, 2)
        self.assertEqual(router.engine._backends[0].total_failures, 0)

        # 3. /v1/completions with usage only on non-terminal chunk still withholds [DONE].
        stream_mode[0] = "legacy_interim_only"
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-legacy-usage",
                    "prompt": "Say hello",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(recorded_usage[2].status_code, 502)

        # 4. vLLM continuous_usage_stats chat stream missing final choices:[] frame still rejects [DONE].
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        stream_mode[0] = "chat_vllm_continuous_missing_final"
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-legacy-usage",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"upstream_stream_interrupted", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(recorded_usage[3].status_code, 502)

    def test_tpu_exporter_https_retries_alternate_address_on_tls_stall_and_cert_error(
        self,
    ) -> None:
        from router import tpu_telemetry as telemetry_mod

        exporter = self._track(_MockTpuExporter(health_status=1, generation="v6e"))
        good_port = exporter.server_address[1]

        # Listener that accepts TCP immediately then stalls forever during TLS handshake.
        stall_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stall_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        stall_srv.bind(("127.0.0.1", 0))
        stall_srv.listen(4)
        stall_port = stall_srv.getsockname()[1]
        accepted_stall: List[socket.socket] = []

        def _accept_and_stall() -> None:
            try:
                c, _ = stall_srv.accept()
                accepted_stall.append(c)
                time.sleep(2.0)
            except OSError:
                pass

        stall_thread = threading.Thread(target=_accept_and_stall, daemon=True)
        stall_thread.start()

        class _PassThroughTLSSock:
            """Test double wrapping a connected socket to verify TLS handshake + cleanup."""

            def __init__(self, raw_sock: socket.socket, *, fail_cert: bool = False) -> None:
                self._raw = raw_sock
                self._fail_cert = fail_cert
                self.handshake_called = False
                self.closed = False

            def do_handshake(self) -> None:
                self.handshake_called = True
                if self._fail_cert:
                    raise ssl.SSLCertVerificationError(1, "certificate verify failed")

            def close(self) -> None:
                self.closed = True
                self._raw.close()

            def __getattr__(self, name: str) -> Any:
                return getattr(self._raw, name)

        real_ssl_ctx = ssl.create_default_context()
        wrapped_calls: List[Tuple[int, Optional[str], bool]] = []
        created_wrappers: List[_PassThroughTLSSock] = []
        mode = ["tls_stall_then_good"]

        class _FailoverTestSSLContext:
            def __getattr__(self, name: str) -> Any:
                return getattr(real_ssl_ctx, name)

            def wrap_socket(
                self,
                sock: socket.socket,
                *,
                server_hostname: Optional[str] = None,
                do_handshake_on_connect: bool = True,
                suppress_ragged_eofs: bool = True,
            ) -> Any:
                peer_port = sock.getpeername()[1]
                wrapped_calls.append((peer_port, server_hostname, do_handshake_on_connect))
                if mode[0] == "tls_stall_then_good" and peer_port == stall_port:
                    return real_ssl_ctx.wrap_socket(
                        sock,
                        server_hostname=server_hostname,
                        do_handshake_on_connect=do_handshake_on_connect,
                        suppress_ragged_eofs=suppress_ragged_eofs,
                    )
                fail_cert = (
                    mode[0] == "all_bad_cert"
                    or (mode[0] == "bad_cert_then_good" and peer_port == stall_port)
                )
                wrapper = _PassThroughTLSSock(sock, fail_cert=fail_cert)
                created_wrappers.append(wrapper)
                return wrapper

        # Python 3.10/3.11 HTTPSConnection.__init__ reads verify_mode and check_hostname.
        probe_ctx = _FailoverTestSSLContext()
        self.assertEqual(probe_ctx.verify_mode, real_ssl_ctx.verify_mode)
        self.assertEqual(probe_ctx.check_hostname, real_ssl_ctx.check_hostname)

        fake_infos = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", stall_port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", good_port),
            ),
        ]
        try:
            # 1. Address 0 accepts TCP then stalls in real TLS handshake -> times out within
            #    attempt budget and falls back to address 1.
            t0 = time.monotonic()
            with (
                mock.patch.object(
                    telemetry_mod, "_resolve_host_with_deadline", return_value=fake_infos
                ),
                mock.patch.object(
                    ssl, "create_default_context", return_value=_FailoverTestSSLContext()
                ),
            ):
                telem = scrape_tpu_exporter(
                    f"https://tls-tpu.local:{good_port}/metrics",
                    timeout_s=0.6,
                )
            elapsed = time.monotonic() - t0
            self.assertTrue(telem.reachable)
            self.assertTrue(telem.exporter_up)
            self.assertEqual(telem.healthy_chips, 1)
            self.assertLess(elapsed, 0.6)
            self.assertEqual(len(wrapped_calls), 2)
            self.assertEqual(wrapped_calls[0], (stall_port, "tls-tpu.local", False))
            self.assertEqual(wrapped_calls[1], (good_port, "tls-tpu.local", False))
            self.assertTrue(created_wrappers[-1].handshake_called)

            # 2. Address 0 fails TLS certificate verification -> closes socket 0 and retries addr 1.
            wrapped_calls.clear()
            created_wrappers.clear()
            mode[0] = "bad_cert_then_good"
            with (
                mock.patch.object(
                    telemetry_mod, "_resolve_host_with_deadline", return_value=fake_infos
                ),
                mock.patch.object(
                    ssl, "create_default_context", return_value=_FailoverTestSSLContext()
                ),
            ):
                telem_cert = scrape_tpu_exporter(
                    f"https://tls-tpu.local:{good_port}/metrics",
                    timeout_s=0.6,
                )
            self.assertTrue(telem_cert.reachable)
            self.assertEqual(telem_cert.healthy_chips, 1)
            self.assertEqual(len(created_wrappers), 2)
            self.assertTrue(created_wrappers[0].closed)
            self.assertTrue(created_wrappers[1].handshake_called)

            # 3. If ALL addresses fail certificate verification, scrape fails closed.
            wrapped_calls.clear()
            created_wrappers.clear()
            mode[0] = "all_bad_cert"
            with (
                mock.patch.object(
                    telemetry_mod, "_resolve_host_with_deadline", return_value=fake_infos
                ),
                mock.patch.object(
                    ssl, "create_default_context", return_value=_FailoverTestSSLContext()
                ),
            ):
                telem_fail = scrape_tpu_exporter(
                    f"https://tls-tpu.local:{good_port}/metrics",
                    timeout_s=0.6,
                )
            self.assertFalse(telem_fail.reachable)
            self.assertTrue(all(w.closed for w in created_wrappers))
        finally:
            for c in accepted_stall:
                try:
                    c.close()
                except OSError:
                    pass
            stall_srv.close()
            stall_thread.join(timeout=2.5)

    def test_sse_json_complexity_and_parse_memory_limits_enforced_pre_and_post_commit(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        deep_json = (b"[" * 2500) + (b"]" * 2500)
        # Dense JSON array of empty dicts: ~9 KiB wire bytes, >64 KiB estimated parse memory.
        dense_json = b'{"x":[' + b",".join([b"{}"] * 3000) + b"]}"
        valid_first = (
            b'data: {"id":"c1","object":"chat.completion.chunk",'
            b'"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        )
        mode = ["pre_deep"]

        class _ComplexSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-sse-complex"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode[0] == "pre_deep":
                    self.wfile.write(b"data: " + deep_json + b"\n\ndata: [DONE]\n\n")
                elif mode[0] == "post_deep":
                    self.wfile.write(valid_first)
                    self.wfile.flush()
                    self.wfile.write(b"data: " + deep_json + b"\n\ndata: [DONE]\n\n")
                elif mode[0] == "pre_dense":
                    self.wfile.write(b"data: " + dense_json + b"\n\ndata: [DONE]\n\n")
                else:
                    self.wfile.write(valid_first)
                    self.wfile.flush()
                    self.wfile.write(b"data: " + dense_json + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _ComplexSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_total_response_memory_bytes=64 * 1024,
            max_sse_pre_commit_bytes=32 * 1024,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="sse-complex-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-sse-complex"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Pre-commit deeply nested SSE JSON returns clean 502 (no RecursionError / drop).
        st1, _, bd1 = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse-complex",
                "stream": True,
                "messages": [{"role": "user", "content": "pre-deep"}],
            },
        )
        self.assertEqual(st1, 502)
        self.assertIn(b"response_too_large", bd1)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

        # 2. Post-commit deeply nested SSE JSON emits response_too_large and withholds [DONE].
        mode[0] = "post_deep"
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-sse-complex",
                    "stream": True,
                    "messages": [{"role": "user", "content": "post-deep"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"response_too_large", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(recorded_usage[-1].status_code, 502)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

        # 3. Pre-commit dense SSE JSON exceeding parse memory budget returns 503 without cooldown.
        mode[0] = "pre_dense"
        st3, _, bd3 = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-sse-complex",
                "stream": True,
                "messages": [{"role": "user", "content": "pre-dense"}],
            },
        )
        self.assertEqual(st3, 503)
        self.assertIn(b"server_memory_limit_reached", bd3)
        self.assertEqual(router.engine._backends[0].total_failures, 0)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

        # 4. Post-commit dense SSE JSON exceeding parse memory budget emits 503 error without cooldown.
        mode[0] = "post_dense"
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-sse-complex",
                    "stream": True,
                    "messages": [{"role": "user", "content": "post-dense"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as exc_info:
                resp.read()
            self.assertIn(b"server_memory_limit_reached", exc_info.exception.partial)
            self.assertNotIn(b"data: [DONE]", exc_info.exception.partial)
        finally:
            conn.close()
        self.assertEqual(recorded_usage[-1].status_code, 503)
        self.assertEqual(router.engine._backends[0].total_failures, 0)
        self.assertEqual(router.engine._resp_mem_used_bytes, 0)

    def test_blocking_sse_path_excludes_downstream_write_delay_from_upstream_timeout(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _TwoChunkSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-slow-client"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"sc1","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"First"},"finish_reason":null}]}\n\n'
                )
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(
                    b'data: {"id":"sc1","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"Second"},"finish_reason":null}]}\n\n'
                )
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(
                    b'data: {"id":"sc1","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"Done"},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _TwoChunkSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            request_timeout_s=0.25,
            client_read_timeout_s=1.2,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="slow-client-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-slow-client"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]
        # Ensure serve_forever() has started sse_reactor before patching it to None.
        st_h, _, _ = _http_request(rport, "GET", "/health")
        self.assertEqual(st_h, 200)

        orig_send = socket.socket.send
        delayed_once = [False]
        stall_mode = [False]

        def _slow_downstream_send(sock_self: socket.socket, data: Any, *args: Any) -> int:
            try:
                is_router_client_sock = sock_self.getsockname()[1] == rport
            except OSError:
                is_router_client_sock = False
            if is_router_client_sock:
                raw = bytes(data)
                if b"Second" in raw:
                    if stall_mode[0]:
                        raise TimeoutError("simulated client write timeout")
                    if not delayed_once[0]:
                        delayed_once[0] = True
                        # Delay exceeds backend request_timeout_s (0.25s) but fits client_read_timeout_s (1.2s).
                        time.sleep(0.35)
            return orig_send(sock_self, data, *args)

        # Force the blocking Phase 2 SSE path (same path used for HTTPS backends).
        with mock.patch.object(router, "sse_reactor", None):
            # 1. Slow client write > backend timeout must NOT truncate stream or cool down backend.
            with mock.patch.object(socket.socket, "send", new=_slow_downstream_send):
                st_ok, _, bd_ok = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-slow-client",
                        "stream": True,
                        "messages": [{"role": "user", "content": "slow-ok"}],
                    },
                )
            self.assertTrue(delayed_once[0])
            self.assertEqual(st_ok, 200)
            self.assertIn(b"First", bd_ok)
            self.assertIn(b"Second", bd_ok)
            self.assertIn(b"data: [DONE]", bd_ok)
            self.assertEqual(recorded_usage[-1].status_code, 200)
            self.assertEqual(router.engine._backends[0].total_failures, 0)
            self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

            # 2. Client write timeout records 499 without penalizing the healthy backend.
            stall_mode[0] = True
            with mock.patch.object(socket.socket, "send", new=_slow_downstream_send):
                conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
                try:
                    body = json.dumps(
                        {
                            "model": "m-slow-client",
                            "stream": True,
                            "messages": [{"role": "user", "content": "slow-timeout"}],
                        }
                    ).encode("utf-8")
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        body=body,
                        headers={"Content-Type": "application/json"},
                    )
                    resp = conn.getresponse()
                    self.assertEqual(resp.status, 200)
                    with self.assertRaises(http.client.IncompleteRead):
                        resp.read()
                finally:
                    conn.close()
        deadline = time.monotonic() + 1.0
        while len(recorded_usage) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(recorded_usage), 2)
        self.assertEqual(recorded_usage[1].status_code, 499)
        self.assertEqual(router.engine._backends[0].total_failures, 0)
        self.assertEqual(router.engine._backends[0].cooldown_until_monotonic, 0.0)

    def test_streamed_batch_completions_accumulates_per_choice_usage_and_prefers_request_wide_total(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        include_request_wide_frame = [False]

        class _BatchCompletionsSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-batch-usage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"b1","object":"text_completion",'
                    b'"choices":[{"index":0,"text":"A1","finish_reason":null}]}\n\n'
                    b'data: {"id":"b1","object":"text_completion",'
                    b'"choices":[{"index":1,"text":"A2","finish_reason":null}]}\n\n'
                    b'data: {"id":"b1","object":"text_completion",'
                    b'"choices":[{"index":0,"text":"","finish_reason":"stop"}],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
                    b'data: {"id":"b1","object":"text_completion",'
                    b'"choices":[{"index":1,"text":"","finish_reason":"stop"}],'
                    b'"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}\n\n'
                )
                if include_request_wide_frame[0]:
                    self.wfile.write(
                        b'data: {"id":"b1","object":"text_completion","choices":[],'
                        b'"usage":{"prompt_tokens":25,"completion_tokens":14,"total_tokens":39}}\n\n'
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _BatchCompletionsSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="batch-usage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-batch-usage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Per-choice terminal usage chunks accumulate across batch choices.
        st1, _, bd1 = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-batch-usage",
                "prompt": ["Q1", "Q2"],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(st1, 200)
        self.assertIn(b"data: [DONE]", bd1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 22)
        self.assertEqual(recorded_usage[0].completion_tokens, 12)

        # 2. Request-wide empty-choices usage frame takes precedence when provided.
        include_request_wide_frame[0] = True
        st2, _, bd2 = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-batch-usage",
                "prompt": ["Q1", "Q2"],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(st2, 200)
        self.assertIn(b"data: [DONE]", bd2)
        self.assertEqual(recorded_usage[1].status_code, 200)
        self.assertEqual(recorded_usage[1].prompt_tokens, 25)
        self.assertEqual(recorded_usage[1].completion_tokens, 14)

    def test_model_discovery_preserves_llama_server_aliases_for_routing(self) -> None:
        class _AliasModelsHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "qwen3-32b.gguf",
                                "object": "model",
                                "aliases": ["qwen3-32b", "default", "qwen3-32b.gguf"],
                            }
                        ],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(clen).decode("utf-8"))
                body = json.dumps(
                    {
                        "id": "alias-1",
                        "object": "chat.completion",
                        "model": req.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _AliasModelsHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="llama-alias-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["qwen3-32b"],
                    discover_models=True,
                )
            ],
        )
        router = self._track(create_router_server(cfg, probe_on_start=True))
        rport = router.server_address[1]
        self.assertEqual(
            router.engine._backends[0].effective_models(),
            ["qwen3-32b.gguf", "qwen3-32b", "default"],
        )

        for requested_model in ("qwen3-32b.gguf", "qwen3-32b", "default"):
            st, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": requested_model,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st, 200, f"expected alias {requested_model!r} to route")
            self.assertEqual(hdrs.get("x-freetpu-backend"), "llama-alias-b")

    def test_bearer_auth_rejects_zero_resolved_tokens_at_check_config_and_hook_init(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty bearer token"):
            BearerTokenAuthHook([])
        with self.assertRaisesRegex(ValueError, "non-empty bearer token"):
            BearerTokenAuthHook(["   "])
        with self.assertRaisesRegex(ValueError, "non-empty bearer token"):
            BearerTokenAuthHook({})

        env_key = "FREETPU_TEST_EMPTY_BEARER_TOKEN_ENV"
        prev_val = os.environ.pop(env_key, None)
        cfg_path = Path(self._tmpdir.name) / "empty_bearer_env.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "auth": {"type": "bearer", "tokens_env": [env_key]},
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            # Unset env var fails build_hooks_from_config and --check-config.
            with self.assertRaisesRegex(ValueError, "zero non-empty tokens"):
                build_hooks_from_config(load_config(cfg_path))
            self.assertEqual(
                router_main(["--check-config", "--config", str(cfg_path)]),
                1,
            )

            # Blank-whitespace env var also fails build_hooks_from_config and --check-config.
            os.environ[env_key] = "   \t  "
            with self.assertRaisesRegex(ValueError, "zero non-empty tokens"):
                build_hooks_from_config(load_config(cfg_path))
            self.assertEqual(
                router_main(["--check-config", "--config", str(cfg_path)]),
                1,
            )
        finally:
            if prev_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev_val

    def test_streamed_multi_choice_chat_accumulates_multiple_empty_choice_usage_frames(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _MultiChoiceUsageSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-n2-usage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # llama-server n=2 chat stream: separate choices:[] usage frame after each choice.
                self.wfile.write(
                    b'data: {"id":"c-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"First"},"finish_reason":"stop"}]}\n\n'
                    b'data: {"id":"c-n2","object":"chat.completion.chunk","choices":[],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}}\n\n'
                    b'data: {"id":"c-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":1,"delta":{"content":"Second"},"finish_reason":"stop"}]}\n\n'
                    b'data: {"id":"c-n2","object":"chat.completion.chunk","choices":[],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":9,"total_tokens":19}}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _MultiChoiceUsageSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="n2-usage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-n2-usage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        st, _, bd = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-n2-usage",
                "n": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "two choices"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(b"data: [DONE]", bd)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        # Single shared prompt counted once (10); completion tokens summed across both choices (6+9=15).
        self.assertEqual(recorded_usage[0].prompt_tokens, 10)
        self.assertEqual(recorded_usage[0].completion_tokens, 15)

    def test_fd_headroom_admission_cap_and_emfile_returns_503_without_cooldown(
        self,
    ) -> None:
        import errno
        from router import config as config_mod
        from router import freetpu_router as router_mod

        self.assertEqual(config_mod.DEFAULT_MAX_CONCURRENT_CONNECTIONS, 256)
        if resource is not None and hasattr(resource, "RLIMIT_NOFILE"):
            with mock.patch.object(
                resource, "getrlimit", return_value=(1024, 4096)
            ):
                self.assertEqual(
                    router_mod._effective_max_concurrent_connections(512), 480
                )
                self.assertEqual(
                    router_mod._effective_max_concurrent_connections(256), 256
                )
            with mock.patch.object(
                resource, "getrlimit", return_value=(200, 4096)
            ):
                self.assertEqual(
                    router_mod._effective_max_concurrent_connections(256), 68
                )

        primary = self._track(_MockOpenAIBackend("emfile-primary", ["m-emfile"]))
        backup = self._track(_MockOpenAIBackend("emfile-backup", ["m-emfile"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="emfile-primary",
                    local=True,
                    base_url=primary.base_url,
                    models=["m-emfile"],
                ),
                BackendConfig(
                    id="emfile-backup",
                    local=False,
                    base_url=backup.base_url,
                    models=["m-emfile"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        p_port = primary.server_address[1]
        orig_connect = socket.socket.connect

        def _raise_emfile_on_primary(sock_self: socket.socket, address: Any) -> Any:
            if isinstance(address, tuple) and len(address) >= 2 and address[1] == p_port:
                raise OSError(errno.EMFILE, "Too many open files")
            return orig_connect(sock_self, address)

        with mock.patch.object(socket.socket, "connect", new=_raise_emfile_on_primary):
            st, hdrs, bd = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-emfile",
                    "messages": [{"role": "user", "content": "emfile"}],
                },
            )
        self.assertEqual(st, 503)
        self.assertIn(b"router_overloaded", bd)
        self.assertEqual(hdrs.get("x-freetpu-failovers"), "0")
        self.assertEqual(len(backup.requests_seen), 0)
        for bstate in router.engine._backends:
            self.assertEqual(bstate.total_failures, 0)
            self.assertEqual(bstate.cooldown_until_monotonic, 0.0)
            self.assertTrue(bstate.healthy)

    def test_streamed_multi_choice_timings_sums_predicted_tokens_across_choices(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _MultiChoiceTimingsSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-n2-timings"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"t-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}],'
                    b'"timings":{"prompt_n":12,"predicted_n":5}}\n\n'
                    b'data: {"id":"t-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":"stop"}],'
                    b'"timings":{"prompt_n":12,"predicted_n":20}}\n\n'
                    b'data: {"id":"t-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":1,"delta":{"content":"c"},"finish_reason":"stop"}],'
                    b'"timings":{"prompt_n":12,"predicted_n":30}}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _MultiChoiceTimingsSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="n2-timings-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-n2-timings"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        st, _, bd = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-n2-timings",
                "n": 2,
                "stream": True,
                "messages": [{"role": "user", "content": "timings"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(b"data: [DONE]", bd)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 12)
        self.assertEqual(recorded_usage[0].completion_tokens, 50)

    def test_poll_read_masks_exclude_pollpri_urgent_events(self) -> None:
        import select
        from router import tpu_telemetry as telemetry_mod

        if not hasattr(select, "poll"):
            return
        pollpri = getattr(select, "POLLPRI", 0)
        self.assertEqual(telemetry_mod._POLL_READ_EVENTS & pollpri, 0)
        self.assertNotEqual(telemetry_mod._POLL_READ_EVENTS & select.POLLIN, 0)

        seen_masks: List[int] = []
        real_poll_factory = select.poll

        class _RecordingPoller:
            def __init__(self) -> None:
                self._inner = real_poll_factory()

            def register(self, fd: Any, eventmask: int = 0) -> None:
                seen_masks.append(eventmask)
                self._inner.register(fd, eventmask)

            def modify(self, fd: Any, eventmask: int) -> None:
                seen_masks.append(eventmask)
                self._inner.modify(fd, eventmask)

            def poll(self, timeout: Optional[int] = None) -> List[Tuple[int, int]]:
                return self._inner.poll(timeout)

        s1, s2 = socket.socketpair()
        c1, c2 = socket.socketpair()
        try:
            with mock.patch.object(select, "poll", side_effect=_RecordingPoller):
                telemetry_mod.wait_sockets_readable([s1], 0.0)
                telemetry_mod._wait_io_or_client(s1, c1, wait_write=False, timeout_s=0.0)
                telemetry_mod._wait_io_or_client(s1, c1, wait_write=True, timeout_s=0.0)
            self.assertGreaterEqual(len(seen_masks), 4)
            for mask in seen_masks:
                self.assertEqual(mask & pollpri, 0)
        finally:
            s1.close()
            s2.close()
            c1.close()
            c2.close()

    def test_completions_echo_true_excludes_echoed_prompt_from_fallback_completion_cost(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        long_prompt = "P" * 400
        mode = ["stream_echo_only"]

        class _EchoCompletionsHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-echo"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                if mode[0] == "non_stream_echo_only":
                    body = json.dumps(
                        {
                            "id": "cmpl-echo-ns",
                            "object": "text_completion",
                            "model": "m-echo",
                            "choices": [
                                {"index": 0, "text": long_prompt, "finish_reason": "length"}
                            ],
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode[0] == "stream_echo_only":
                    ev = json.dumps(
                        {
                            "id": "cmpl-echo-1",
                            "object": "text_completion",
                            "choices": [
                                {"index": 0, "text": long_prompt, "finish_reason": "length"}
                            ],
                        }
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + ev + b"\n\ndata: [DONE]\n\n")
                elif mode[0] == "stream_echo_plus_suffix":
                    ev1 = json.dumps(
                        {
                            "id": "cmpl-echo-2",
                            "object": "text_completion",
                            "choices": [
                                {"index": 0, "text": long_prompt, "finish_reason": None}
                            ],
                        }
                    ).encode("utf-8")
                    ev2 = json.dumps(
                        {
                            "id": "cmpl-echo-2",
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": " generated reply",
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode("utf-8")
                    self.wfile.write(
                        b"data: " + ev1 + b"\n\ndata: " + ev2 + b"\n\ndata: [DONE]\n\n"
                    )
                else:
                    ev = json.dumps(
                        {
                            "id": "cmpl-echo-3",
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": long_prompt + " generated reply",
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 95,
                                "completion_tokens": 7,
                                "total_tokens": 102,
                            },
                        }
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + ev + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _EchoCompletionsHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="echo-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-echo"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Echo-only streaming scoring (no usage/timings): 0 completion tokens.
        st1, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo",
                "prompt": long_prompt,
                "echo": True,
                "max_tokens": 0,
                "stream": True,
            },
        )
        self.assertEqual(st1, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 100)
        self.assertEqual(recorded_usage[0].completion_tokens, 0)

        # 2. Echo + 16-char suffix streaming (no usage/timings): only suffix counted (4 tokens).
        mode[0] = "stream_echo_plus_suffix"
        st2, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo",
                "prompt": long_prompt,
                "echo": True,
                "stream": True,
            },
        )
        self.assertEqual(st2, 200)
        self.assertEqual(recorded_usage[1].completion_tokens, 4)

        # 3. Explicit upstream usage still takes precedence over echo-adjusted fallback.
        mode[0] = "stream_echo_explicit_usage"
        st3, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo",
                "prompt": long_prompt,
                "echo": True,
                "stream": True,
            },
        )
        self.assertEqual(st3, 200)
        self.assertEqual(recorded_usage[2].prompt_tokens, 95)
        self.assertEqual(recorded_usage[2].completion_tokens, 7)

        # 4. Non-streaming echo-only scoring (no usage/timings): 0 completion tokens.
        mode[0] = "non_stream_echo_only"
        st4, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo",
                "prompt": long_prompt,
                "echo": True,
                "max_tokens": 0,
                "stream": False,
            },
        )
        self.assertEqual(st4, 200)
        self.assertEqual(recorded_usage[3].completion_tokens, 0)

    def test_loopback_credential_guard_rejects_trailing_dot_ip_and_non_loopback_peer(
        self,
    ) -> None:
        from router.config import is_loopback_host
        from router.freetpu_router import _verify_connected_peer_for_credentials

        self.assertFalse(is_loopback_host("127.0.0.1."))
        self.assertFalse(is_loopback_host("::1."))
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("localhost."))

        with self.assertRaisesRegex(ValueError, "plaintext HTTP"):
            BackendConfig(
                id="dot-ip",
                base_url="http://127.0.0.1.:8000/v1",
                api_key="secret-token",
            ).validate()

        closed = [False]

        class _FakeRemoteSock:
            def getpeername(self) -> Tuple[str, int]:
                return ("203.0.113.10", 8000)

        class _FakeConn:
            def __init__(self) -> None:
                self.sock: Any = _FakeRemoteSock()

            def connect(self) -> None:
                pass

            def close(self) -> None:
                closed[0] = True
                self.sock = None

        bcfg = BackendConfig(
            id="spoofed-localhost",
            base_url="http://localhost:8000/v1",
            api_key="secret-token",
        )
        bcfg.validate()
        with self.assertRaisesRegex(ValueError, "non-loopback peer"):
            _verify_connected_peer_for_credentials(
                _FakeConn(),  # type: ignore[arg-type]
                bcfg,
                "http",
                "secret-token",
            )
        self.assertTrue(closed[0])

    def test_interrupted_multi_choice_continuous_usage_preserves_per_choice_totals(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        complete_with_final_frame = [False]

        class _ContinuousMultiChoiceHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-cont-n2"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b'data: {"id":"v-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n'
                    b'data: {"id":"v-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
                    b'data: {"id":"v-n2","object":"chat.completion.chunk",'
                    b'"choices":[{"index":1,"delta":{"content":"c"},"finish_reason":null}],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":15,"total_tokens":25}}\n\n'
                )
                if complete_with_final_frame[0]:
                    self.wfile.write(
                        b'data: {"id":"v-n2","object":"chat.completion.chunk",'
                        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                        b'data: {"id":"v-n2","object":"chat.completion.chunk",'
                        b'"choices":[{"index":1,"delta":{},"finish_reason":"stop"}]}\n\n'
                        b'data: {"id":"v-n2","object":"chat.completion.chunk","choices":[],'
                        b'"usage":{"prompt_tokens":10,"completion_tokens":40,"total_tokens":50}}\n\n'
                        b"data: [DONE]\n\n"
                    )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _ContinuousMultiChoiceHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="cont-n2-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-cont-n2"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Interrupted multi-choice continuous-usage stream sums latest count per choice (20+15=35).
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            body = json.dumps(
                {
                    "model": "m-cont-n2",
                    "n": 2,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": "interrupted"}],
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            with self.assertRaises(http.client.IncompleteRead):
                resp.read()
        finally:
            conn.close()
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)
        self.assertEqual(recorded_usage[0].prompt_tokens, 10)
        self.assertEqual(recorded_usage[0].completion_tokens, 35)

        # 2. Final request-wide choices:[] usage frame supersedes per-choice sum when it arrives.
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        complete_with_final_frame[0] = True
        st2, _, bd2 = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-cont-n2",
                "n": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "complete"}],
            },
        )
        self.assertEqual(st2, 200)
        self.assertIn(b"data: [DONE]", bd2)
        self.assertEqual(recorded_usage[1].status_code, 200)
        self.assertEqual(recorded_usage[1].prompt_tokens, 10)
        self.assertEqual(recorded_usage[1].completion_tokens, 40)

    def test_tpu_exporter_handles_dns_worker_thread_start_failure_cleanly(self) -> None:
        from router import tpu_telemetry as telemetry_mod

        sem_before = telemetry_mod._DNS_RESOLVER_SEMAPHORE._value
        with mock.patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("can't start new thread"),
        ):
            telem = scrape_tpu_exporter(
                "http://tpu-node.local:9400/metrics",
                timeout_s=0.5,
            )
        self.assertFalse(telem.reachable)
        self.assertFalse(telem.exporter_up)
        self.assertEqual(telem.error, "exporter_network_error:OSError")
        self.assertEqual(telemetry_mod._DNS_RESOLVER_SEMAPHORE._value, sem_before)

    def test_blank_bind_host_rejected_in_config_and_cli_override(self) -> None:
        base_backend = BackendConfig(
            id="b-host",
            base_url="http://127.0.0.1:8081/v1",
            models=["m1"],
        )
        for bad_host in ("", "   ", "\t\n", "[]", "127.0.0.1 extra"):
            with self.assertRaisesRegex(ValueError, "Invalid bind host"):
                RouterConfig(host=bad_host, backends=[base_backend])

        for json_host in ("", "   ", None):
            bad_cfg_path = Path(self._tmpdir.name) / f"bad_host_{repr(json_host)}.json"
            bad_cfg_path.write_text(
                json.dumps(
                    {
                        "host": json_host,
                        "backends": [
                            {
                                "id": "b1",
                                "base_url": "http://127.0.0.1:8081/v1",
                                "models": ["m1"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid bind host"):
                load_config(bad_cfg_path)
            self.assertEqual(
                router_main(["--check-config", "--config", str(bad_cfg_path)]),
                1,
            )

        valid_cfg_path = Path(self._tmpdir.name) / "valid_host_cli_override.json"
        valid_cfg_path.write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            router_main(["--check-config", "--config", str(valid_cfg_path), "--host", ""]),
            1,
        )
        self.assertEqual(
            router_main(["--check-config", "--config", str(valid_cfg_path), "--host", "   "]),
            1,
        )

    def test_interrupted_batched_completions_deduplicates_prompt_tokens_by_prompt_index(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        include_prompt1_choice = [False]

        class _BatchedN2InterruptedHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-batch-n2"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Choices 0 and 1 both belong to prompt 0 (n=2); each repeats prompt 0's 15 tokens.
                self.wfile.write(
                    b'data: {"id":"bn2","object":"text_completion",'
                    b'"choices":[{"index":0,"text":"a","finish_reason":null}],'
                    b'"usage":{"prompt_tokens":15,"completion_tokens":6,"total_tokens":21}}\n\n'
                    b'data: {"id":"bn2","object":"text_completion",'
                    b'"choices":[{"index":1,"text":"b","finish_reason":null}],'
                    b'"usage":{"prompt_tokens":15,"completion_tokens":9,"total_tokens":24}}\n\n'
                )
                if include_prompt1_choice[0]:
                    # Choice 2 belongs to prompt 1 (20 prompt tokens, 4 completion tokens).
                    self.wfile.write(
                        b'data: {"id":"bn2","object":"text_completion",'
                        b'"choices":[{"index":2,"text":"c","finish_reason":null}],'
                        b'"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}\n\n'
                    )
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _BatchedN2InterruptedHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="batch-n2-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-batch-n2"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        def _send_interrupted_batch() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
            try:
                body = json.dumps(
                    {
                        "model": "m-batch-n2",
                        "prompt": ["Prompt zero", "Prompt one"],
                        "n": 2,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                ).encode("utf-8")
                conn.request(
                    "POST",
                    "/v1/completions",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                with self.assertRaises(http.client.IncompleteRead):
                    resp.read()
            finally:
                conn.close()

        # 1. Interrupted after choices 0 & 1 (both prompt 0): prompt counted once (15), completions summed (15).
        _send_interrupted_batch()
        self.assertEqual(recorded_usage[0].status_code, 502)
        self.assertEqual(recorded_usage[0].prompt_tokens, 15)
        self.assertEqual(recorded_usage[0].completion_tokens, 15)

        # 2. Interrupted after choices 0, 1 (prompt 0) & 2 (prompt 1): prompts 15+20=35, completions 6+9+4=19.
        router.engine._backends[0].healthy = True
        router.engine._backends[0].cooldown_until_monotonic = 0.0
        include_prompt1_choice[0] = True
        _send_interrupted_batch()
        self.assertEqual(recorded_usage[1].status_code, 502)
        self.assertEqual(recorded_usage[1].prompt_tokens, 35)
        self.assertEqual(recorded_usage[1].completion_tokens, 19)

    def test_tpu_exporter_handles_chunk_size_overflow_error_cleanly(self) -> None:
        exporter = self._track(_MockTpuExporter(health_status=1, generation="v6e"))
        with mock.patch.object(
            http.client.HTTPResponse,
            "read1",
            side_effect=OverflowError("Python int too large to convert to C ssize_t"),
        ):
            telem = scrape_tpu_exporter(
                f"http://127.0.0.1:{exporter.server_address[1]}/metrics",
                timeout_s=1.0,
            )
        self.assertFalse(telem.reachable)
        self.assertFalse(telem.exporter_up)
        self.assertEqual(telem.error, "exporter_error:OverflowError")

    def test_unknown_usage_meter_and_scaler_types_rejected_by_validation_and_check_config(
        self,
    ) -> None:
        base_backend = BackendConfig(
            id="b-meter",
            base_url="http://127.0.0.1:8081/v1",
            models=["m1"],
        )
        with self.assertRaisesRegex(ValueError, "Invalid usage_meter.type"):
            RouterConfig(
                backends=[base_backend],
                usage_meter={"type": "jsnol"},
            )
        with self.assertRaisesRegex(ValueError, "jsonl_path"):
            RouterConfig(
                backends=[base_backend],
                usage_meter={"type": "noop", "jsonl_path": "/tmp/usage.jsonl"},
            )
        with self.assertRaisesRegex(ValueError, "Invalid scaler.type"):
            RouterConfig(
                backends=[base_backend],
                scaler={"type": "autoscalerr"},
            )

        bad_cfg_path = Path(self._tmpdir.name) / "bad_usage_meter_type.json"
        bad_cfg_path.write_text(
            json.dumps(
                {
                    "usage_meter": {"type": "jsnol"},
                    "backends": [
                        {
                            "id": "b1",
                            "base_url": "http://127.0.0.1:8081/v1",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            router_main(["--check-config", "--config", str(bad_cfg_path)]),
            1,
        )

    def test_completions_echo_with_return_token_ids_does_not_subtract_suppressed_prompt(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        long_prompt = "Q" * 400
        mode = ["stream_with_token_ids"]

        class _EchoReturnTokenIdsHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-echo-tids"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode[0] == "stream_with_token_ids":
                    # vLLM v0.33.0 echo=true + return_token_ids=true suppresses textual prompt
                    # echo and emits prompt_token_ids inside the streamed choice alongside token_ids.
                    ev = json.dumps(
                        {
                            "id": "cmpl-tids-1",
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "Hi!",
                                    "prompt_token_ids": [1, 2, 3, 4],
                                    "token_ids": [9906, 0, 11],
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode("utf-8")
                elif mode[0] == "stream_text_only_with_return_token_ids":
                    # return_token_ids=true without choice.token_ids still skips echo subtraction.
                    ev = json.dumps(
                        {
                            "id": "cmpl-tids-2",
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "Short answer",
                                    "prompt_token_ids": [1, 2, 3, 4],
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode("utf-8")
                else:
                    # echo=true with return_token_ids=false: vLLM echoes the prompt in text AND
                    # may still emit prompt_token_ids; echoed prompt MUST still be subtracted.
                    ev = json.dumps(
                        {
                            "id": "cmpl-tids-3",
                            "object": "text_completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": long_prompt + "Short answer",
                                    "prompt_token_ids": [1, 2, 3, 4],
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode("utf-8")
                self.wfile.write(b"data: " + ev + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _EchoReturnTokenIdsHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="echo-tids-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-echo-tids"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # Choice-level prompt_token_ids alone (no text/token_ids/finish_reason) still buffers pre-commit.
        self.assertFalse(
            _sse_event_should_commit(
                json.dumps(
                    {
                        "id": "cmpl-buf",
                        "object": "text_completion",
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "prompt_token_ids": [1, 2, 3, 4],
                                "finish_reason": None,
                            }
                        ],
                    }
                ).encode("utf-8")
            )
        )

        # 1. With return_token_ids=true + choice.token_ids: exact generated token count (3) recorded.
        st1, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo-tids",
                "prompt": long_prompt,
                "echo": True,
                "return_token_ids": True,
                "stream": True,
            },
        )
        self.assertEqual(st1, 200)
        self.assertEqual(recorded_usage[0].completion_tokens, 3)

        # 2. With return_token_ids=true and no choice.token_ids: short non-echoed text (12 chars -> 3 tokens)
        # is NOT subtracted to 0.
        mode[0] = "stream_text_only_with_return_token_ids"
        st2, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo-tids",
                "prompt": long_prompt,
                "echo": True,
                "return_token_ids": True,
                "stream": True,
            },
        )
        self.assertEqual(st2, 200)
        self.assertEqual(recorded_usage[1].completion_tokens, 3)

        # 3. With echo=true and return_token_ids=false: even when prompt_token_ids is present,
        # echoed 400-char prompt IS subtracted so only the 12-char completion (3 tokens) is billed.
        mode[0] = "stream_echoed_text_with_prompt_token_ids"
        st3, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo-tids",
                "prompt": long_prompt,
                "echo": True,
                "return_token_ids": False,
                "stream": True,
            },
        )
        self.assertEqual(st3, 200)
        self.assertEqual(recorded_usage[2].completion_tokens, 3)

    def test_sse_excess_empty_choice_usage_frames_rejected_without_progress_renewal(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _RunawayUsageHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-runaway-usage"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                tok = (
                    b'data: {"id":"c1","object":"chat.completion.chunk","choices":'
                    b'[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
                )
                usage_ev = (
                    b'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
                    b'"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
                )
                self.wfile.write(tok + usage_ev)
                self.wfile.flush()
                try:
                    for _ in range(50):
                        self.wfile.write(usage_ev)
                        self.wfile.flush()
                        time.sleep(0.01)
                except OSError:
                    pass

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _RunawayUsageHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            request_timeout_s=2.0,
            backends=[
                BackendConfig(
                    id="runaway-usage-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-runaway-usage"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        t0 = time.monotonic()
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "model": "m-runaway-usage",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            raw_stream = b""
            try:
                raw_stream = resp.read()
            except http.client.IncompleteRead as exc:
                raw_stream = exc.partial
        finally:
            conn.close()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 1.5)
        self.assertEqual(raw_stream.count(b'"usage"'), 1)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 502)
        self.assertEqual(recorded_usage[0].prompt_tokens, 10)
        self.assertEqual(recorded_usage[0].completion_tokens, 5)
        self.assertEqual(router.engine._backends[0].total_failures, 1)

    def test_dual_stack_ipv4_mapped_ipv6_listener_accepts_ipv4_host_and_peer_loopback(
        self,
    ) -> None:
        from email.message import Message
        from router.config import is_loopback_host
        from router.freetpu_router import (
            RouterEngine,
            _RouterRequestHandler,
            _verify_connected_peer_for_credentials,
        )

        self.assertTrue(is_loopback_host("::ffff:127.0.0.1"))
        self.assertFalse(is_loopback_host("::ffff:192.0.2.10"))

        class _FakeSock:
            def __init__(self, local_ip: str, peer_ip: str) -> None:
                self._local_ip = local_ip
                self._peer_ip = peer_ip

            def getsockname(self) -> Tuple[str, int, int, int]:
                return (self._local_ip, 8080, 0, 0)

            def getpeername(self) -> Tuple[str, int, int, int]:
                return (self._peer_ip, 54321, 0, 0)

        class _FakeConn:
            def __init__(self, peer_ip: str) -> None:
                self.sock = _FakeSock("::ffff:127.0.0.1", peer_ip)
                self.closed = False

            def close(self) -> None:
                self.closed = True

        cfg = RouterConfig(
            host="::",
            port=8080,
            backends=[
                BackendConfig(
                    id="b-dual",
                    type="pixel-tpu",
                    local=True,
                    base_url="http://127.0.0.1:8081/v1",
                    models=["m1"],
                )
            ],
        )
        engine = RouterEngine(cfg)
        fake_server = mock.MagicMock()
        fake_server.configured_bind_host = "::"
        fake_server.server_address = ("::", 8080, 0, 0)
        fake_server.engine = engine

        handler = object.__new__(_RouterRequestHandler)
        handler.server = fake_server
        handler.connection = _FakeSock("::ffff:192.0.2.10", "::ffff:192.0.2.50")

        hdrs_ok = Message()
        hdrs_ok["Host"] = "192.0.2.10:8080"
        handler.headers = hdrs_ok
        self.assertTrue(handler._validate_request_host())

        hdrs_bad = Message()
        hdrs_bad["Host"] = "192.0.2.99:8080"
        handler.headers = hdrs_bad
        self.assertFalse(handler._validate_request_host())

        # Peer loopback check unwraps IPv4-mapped IPv6 addresses.
        backend = cfg.backends[0]
        _verify_connected_peer_for_credentials(
            _FakeConn("::ffff:127.0.0.1"),  # type: ignore[arg-type]
            backend,
            "http",
            "secret-key",
        )
        bad_conn = _FakeConn("::ffff:192.0.2.10")
        with self.assertRaises(ValueError):
            _verify_connected_peer_for_credentials(
                bad_conn,  # type: ignore[arg-type]
                backend,
                "http",
                "secret-key",
            )
        self.assertTrue(bad_conn.closed)

    def test_cors_exposes_router_response_headers_for_allowed_origins(self) -> None:
        mock_backend = self._track(
            _MockOpenAIBackend("cors-exp-b", ["m-cors-exp"])
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            cors_allowed_origins=["https://app.example.com"],
            backends=[
                BackendConfig(
                    id="cors-exp-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=mock_backend.base_url,
                    models=["m-cors-exp"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        expected_exposed = "X-FreeTPU-Backend, X-FreeTPU-Failovers, X-FreeTPU-Request-Id"

        st_ns, hdrs_ns, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://app.example.com"},
            body={
                "model": "m-cors-exp",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        self.assertEqual(st_ns, 200)
        self.assertEqual(hdrs_ns.get("access-control-allow-origin"), "https://app.example.com")
        self.assertEqual(hdrs_ns.get("access-control-expose-headers"), expected_exposed)

        st_s, hdrs_s, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            headers={"Origin": "https://app.example.com"},
            body={
                "model": "m-cors-exp",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        self.assertEqual(st_s, 200)
        self.assertEqual(hdrs_s.get("access-control-allow-origin"), "https://app.example.com")
        self.assertEqual(hdrs_s.get("access-control-expose-headers"), expected_exposed)

    def test_prometheus_compact_labeled_sample_without_space_after_brace(self) -> None:
        compact_metrics = "\n".join(
            [
                "freetpu_exporter_up 1",
                "freetpu_exporter_up1",
                'freetpu_tpu_devices_discovered 1',
                'freetpu_tpu_health_status{chip="0"}1',
                'freetpu_tpu_hbm_utilization_ratio{chip="0",generation="v5e"}0.42 1700000000000',
            ]
        ) + "\n"
        parsed = parse_prometheus_text(compact_metrics)
        by_name = {s.name: s for s in parsed}
        self.assertIn("freetpu_tpu_health_status", by_name)
        self.assertEqual(by_name["freetpu_tpu_health_status"].labels, {"chip": "0"})
        self.assertEqual(by_name["freetpu_tpu_health_status"].value, 1.0)
        self.assertIn("freetpu_tpu_hbm_utilization_ratio", by_name)
        self.assertAlmostEqual(by_name["freetpu_tpu_hbm_utilization_ratio"].value, 0.42)
        self.assertNotIn("freetpu_exporter_up1", by_name)

        exporter = self._track(_MockTpuExporter(raw_body=compact_metrics))
        telemetry = scrape_tpu_exporter(
            exporter.metrics_url,
            timeout_s=2.0,
        )
        self.assertTrue(telemetry.reachable)
        self.assertTrue(telemetry.exporter_up)
        self.assertEqual(telemetry.healthy_chips, 1)
        self.assertEqual(telemetry.total_chips, 1)
        self.assertAlmostEqual(telemetry.max_hbm_utilization_ratio or 0.0, 0.42)
        ready, reason = evaluate_tpu_readiness(telemetry)
        self.assertTrue(ready, msg=f"unexpected unready reason: {reason}")

    def test_large_batch_over_1024_choices_preserves_final_usage_and_done(self) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _LargeBatchSSEHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-batch-1200"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Emit 1,200 terminal choices (600 prompts x n=2) followed by one aggregate usage frame + [DONE].
                parts: List[bytes] = []
                for start in range(0, 1200, 100):
                    ev = json.dumps(
                        {
                            "id": "cmpl-1200",
                            "object": "text_completion",
                            "choices": [
                                {"index": idx, "text": "x", "finish_reason": "stop"}
                                for idx in range(start, start + 100)
                            ],
                        }
                    ).encode("utf-8")
                    parts.append(b"data: " + ev + b"\n\n")
                usage_ev = json.dumps(
                    {
                        "id": "cmpl-1200",
                        "object": "text_completion",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 6000,
                            "completion_tokens": 2400,
                            "total_tokens": 8400,
                        },
                    }
                ).encode("utf-8")
                parts.append(b"data: " + usage_ev + b"\n\n")
                parts.append(b"data: [DONE]\n\n")
                self.wfile.write(b"".join(parts))
                self.wfile.flush()

        srv = self._track(ThreadingHTTPServer(("127.0.0.1", 0), _LargeBatchSSEHandler))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="batch-1200-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-batch-1200"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        st, _, bd = _http_request(
            router.server_address[1],
            "POST",
            "/v1/completions",
            body={
                "model": "m-batch-1200",
                "prompt": [f"p{i}" for i in range(600)],
                "n": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(st, 200)
        self.assertIn(b"data: [DONE]", bd)
        self.assertEqual(len(recorded_usage), 1)
        self.assertEqual(recorded_usage[0].status_code, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 6000)
        self.assertEqual(recorded_usage[0].completion_tokens, 2400)
        bstate = router.engine._backends[0]
        self.assertEqual(bstate.total_failures, 0)
        self.assertTrue(bstate.healthy)
        self.assertEqual(bstate.cooldown_until_monotonic, 0.0)

    def test_non_streaming_completion_omits_ttft_histogram_while_streaming_records_it(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        backend = self._track(_MockOpenAIBackend("ttft-sep-b", ["m-ttft-sep"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="ttft-sep-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                    models=["m-ttft-sep"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        st_ns, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ttft-sep",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        self.assertEqual(st_ns, 200)
        self.assertEqual(len(recorded_usage), 1)
        self.assertIsNone(recorded_usage[0].ttft_ms)

        st_m1, _, metrics_1_bytes = _http_request(rport, "GET", "/metrics")
        self.assertEqual(st_m1, 200)
        metrics_1 = metrics_1_bytes.decode("utf-8")
        self.assertIn(
            'freetpu_router_request_duration_seconds_count{model="m-ttft-sep",backend_id="ttft-sep-b"} 1',
            metrics_1,
        )
        self.assertNotIn(
            'freetpu_router_ttft_seconds_count{model="m-ttft-sep",backend_id="ttft-sep-b"}',
            metrics_1,
        )

        st_s, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-ttft-sep",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        self.assertEqual(st_s, 200)
        self.assertEqual(len(recorded_usage), 2)
        self.assertIsNotNone(recorded_usage[1].ttft_ms)

        st_m2, _, metrics_2_bytes = _http_request(rport, "GET", "/metrics")
        self.assertEqual(st_m2, 200)
        metrics_2 = metrics_2_bytes.decode("utf-8")
        self.assertIn(
            'freetpu_router_request_duration_seconds_count{model="m-ttft-sep",backend_id="ttft-sep-b"} 2',
            metrics_2,
        )
        self.assertIn(
            'freetpu_router_ttft_seconds_count{model="m-ttft-sep",backend_id="ttft-sep-b"} 1',
            metrics_2,
        )

    def test_tpu_exporter_rejects_unterminated_trailing_line_as_incomplete(
        self,
    ) -> None:
        class _TruncatedCloseDelimExporter(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                # Close-delimited response truncated mid-value ("0.9" cut off from "0.97")
                # with no trailing newline.
                body = (
                    b"freetpu_exporter_up 1\n"
                    b"freetpu_tpu_devices_discovered 1\n"
                    b'freetpu_tpu_health_status{chip="0"} 1\n'
                    b'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.9'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

        srv = self._track(
            ThreadingHTTPServer(("127.0.0.1", 0), _TruncatedCloseDelimExporter)
        )
        telem = scrape_tpu_exporter(
            f"http://127.0.0.1:{srv.server_address[1]}/metrics",
            timeout_s=2.0,
        )
        self.assertFalse(telem.reachable)
        self.assertFalse(telem.exporter_up)
        self.assertEqual(telem.error, "exporter_incomplete_response")
        ready, reason = evaluate_tpu_readiness(telem)
        self.assertFalse(ready)
        self.assertEqual(reason, "tpu_exporter_unreachable")

    def test_discovered_model_ids_match_inference_id_validation_policy(self) -> None:
        overlong_path = "/models/" + ("a" * 260) + ".gguf"
        control_alias = "bad\x07alias"
        valid_alias = "qwen3-32b-q4"

        class _OverlongModelDiscoveryHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": overlong_path,
                                "object": "model",
                                "aliases": [control_alias, valid_alias],
                            }
                        ],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 0:
                    self.rfile.read(clen)
                body = json.dumps(
                    {
                        "id": "chatcmpl-valid-alias",
                        "object": "chat.completion",
                        "model": valid_alias,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

        srv = self._track(
            ThreadingHTTPServer(("127.0.0.1", 0), _OverlongModelDiscoveryHandler)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="gguf-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    discover_models=True,
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        st_m, _, body_m = _http_request(rport, "GET", "/v1/models")
        self.assertEqual(st_m, 200)
        advertised = [entry["id"] for entry in json.loads(body_m.decode("utf-8"))["data"]]
        self.assertEqual(advertised, [valid_alias])

        st_ok, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": valid_alias,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st_ok, 200)

        st_bad, _, body_bad = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={
                "model": overlong_path,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st_bad, 400)
        self.assertIn("invalid_model_id", body_bad.decode("utf-8"))

    def test_tpu_exporter_marks_malformed_pressure_metrics_invalid(self) -> None:
        for bad_line, metric_name in (
            ('freetpu_tpu_duty_cycle_percent{chip="0"} N/A\n', "freetpu_tpu_duty_cycle_percent"),
            (
                'freetpu_tpu_hbm_utilization_ratio{chip="0"} garbage\n',
                "freetpu_tpu_hbm_utilization_ratio",
            ),
        ):
            raw = (
                "freetpu_exporter_up 1\n"
                "freetpu_tpu_devices_discovered 1\n"
                'freetpu_tpu_health_status{chip="0"} 1\n'
                + bad_line
            )
            exporter = self._track(_MockTpuExporter(raw_body=raw))
            telem = scrape_tpu_exporter(exporter.metrics_url, timeout_s=2.0)
            self.assertTrue(telem.reachable)
            self.assertFalse(telem.exporter_up)
            self.assertIsNotNone(telem.error)
            self.assertIn(metric_name, telem.error or "")
            ready, reason = evaluate_tpu_readiness(
                telem,
                max_hbm_utilization=0.95,
                max_duty_cycle_percent=80.0,
            )
            self.assertFalse(ready)
            self.assertEqual(reason, "tpu_exporter_invalid_metrics")

    def test_oversized_integer_in_json_request_returns_400_json(self) -> None:
        backend = self._track(_MockOpenAIBackend("bigint-b", ["m-bigint"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="bigint-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=backend.base_url,
                    models=["m-bigint"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        raw_payload = (
            '{"model":"m-bigint","messages":[],"max_tokens":' + ("9" * 5000) + "}"
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", rport, timeout=3.0)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=raw_payload,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            body = resp.read()
        finally:
            conn.close()
        self.assertEqual(resp.status, 400)
        parsed = json.loads(body.decode("utf-8"))
        self.assertIn("error", parsed)
        self.assertEqual(parsed["error"].get("type"), "invalid_request_error")
        self.assertEqual(len(backend.requests_seen), 0)

    def test_missing_backend_required_fields_reported_as_config_errors(self) -> None:
        from router.freetpu_router import main as router_main

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, bad_backends in enumerate(
                (
                    [{"base_url": "http://127.0.0.1:8081/v1"}],
                    [{"id": "missing-url-b"}],
                    ["not-an-object"],
                )
            ):
                with self.assertRaises(ValueError):
                    RouterConfig.from_dict({"backends": bad_backends})
                cfg_path = Path(tmpdir) / f"missing-backend-field-{idx}.json"
                cfg_path.write_text(
                    json.dumps({"backends": bad_backends}),
                    encoding="utf-8",
                )
                err_buf = io.StringIO()
                with mock.patch("sys.stderr", err_buf):
                    rc = router_main(["--check-config", "--config", str(cfg_path)])
                self.assertEqual(rc, 1)
                self.assertIn("Config error:", err_buf.getvalue())

    def test_https_tpu_exporter_disables_ragged_eof_suppression_and_rejects_ssl_eof(
        self,
    ) -> None:
        from router import tpu_telemetry as telemetry_mod

        class _RaggedEOFTLSSock:
            def __init__(self, sock: socket.socket) -> None:
                self._sock = sock
                self._chunks = [
                    (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/plain; version=0.0.4\r\n"
                        b"Connection: close\r\n"
                        b"\r\n"
                        b"freetpu_exporter_up 1\n"
                        b"freetpu_tpu_devices_discovered 1\n"
                        b'freetpu_tpu_health_status{chip="0"} 1\n'
                    )
                ]

            def do_handshake(self) -> None:
                return None

            def settimeout(self, t: Optional[float]) -> None:
                self._sock.settimeout(t)

            def setblocking(self, flag: bool) -> None:
                self._sock.setblocking(flag)

            def gettimeout(self) -> Optional[float]:
                return self._sock.gettimeout()

            def fileno(self) -> int:
                return self._sock.fileno()

            def pending(self) -> int:
                return len(self._chunks[0]) if self._chunks else 0

            def sendall(self, data: bytes) -> None:
                return None

            def recv(self, n: int) -> bytes:
                if self._chunks:
                    chunk = self._chunks.pop(0)
                    return chunk
                raise ssl.SSLEOFError(8, "EOF occurred in violation of protocol")

            def recv_into(self, buf: Any, nbytes: int = 0) -> int:
                data = self.recv(nbytes or len(buf))
                buf[: len(data)] = data
                return len(data)

            def close(self) -> None:
                try:
                    self._sock.close()
                except OSError:
                    pass

        observed_kwargs: List[Dict[str, Any]] = []
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        lport = listener.getsockname()[1]

        def _accept_once() -> None:
            try:
                conn, _ = listener.accept()
                time.sleep(0.05)
                conn.close()
            except OSError:
                pass

        t = threading.Thread(target=_accept_once, daemon=True)
        t.start()

        class _CaptureSSLContext:
            verify_mode = ssl.CERT_REQUIRED
            check_hostname = True

            def wrap_socket(
                self,
                sock: socket.socket,
                *,
                server_hostname: Optional[str] = None,
                do_handshake_on_connect: bool = True,
                suppress_ragged_eofs: bool = True,
            ) -> Any:
                observed_kwargs.append(
                    {
                        "server_hostname": server_hostname,
                        "do_handshake_on_connect": do_handshake_on_connect,
                        "suppress_ragged_eofs": suppress_ragged_eofs,
                    }
                )
                return _RaggedEOFTLSSock(sock)

        try:
            with mock.patch.object(
                telemetry_mod.ssl,
                "create_default_context",
                return_value=_CaptureSSLContext(),
            ):
                telem = scrape_tpu_exporter(
                    f"https://127.0.0.1:{lport}/metrics",
                    timeout_s=2.0,
                )
        finally:
            listener.close()
            t.join(timeout=1.0)

        self.assertEqual(len(observed_kwargs), 1)
        self.assertIs(observed_kwargs[0]["suppress_ragged_eofs"], False)
        self.assertFalse(telem.reachable)
        self.assertFalse(telem.exporter_up)
        self.assertEqual(telem.error, "exporter_tls_error")
        ready, reason = evaluate_tpu_readiness(telem)
        self.assertFalse(ready)
        self.assertEqual(reason, "tpu_exporter_unreachable")

    def test_tpu_exporter_validates_duplicate_conflicting_and_malformed_content_length(
        self,
    ) -> None:
        valid_body = (
            b"freetpu_exporter_up 1\n"
            b"freetpu_tpu_devices_discovered 1\n"
            b'freetpu_tpu_health_status{chip="0"} 1\n'
            b'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
        )
        mode = ["dup_truncated"]

        class _ContentLengthExporter(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                if mode[0] == "dup_truncated":
                    self.send_header("Content-Length", "1200, 1200")
                elif mode[0] == "dup_exact":
                    exact = str(len(valid_body))
                    self.send_header("Content-Length", exact)
                    self.send_header("Content-Length", exact)
                elif mode[0] == "conflicting":
                    self.send_header("Content-Length", "100, 200")
                else:
                    self.send_header("Content-Length", "abc")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(valid_body)
                self.wfile.flush()

        srv = self._track(
            ThreadingHTTPServer(("127.0.0.1", 0), _ContentLengthExporter)
        )
        url = f"http://127.0.0.1:{srv.server_address[1]}/metrics"

        # 1. Duplicate matching Content-Length truncated early -> incomplete scrape.
        telem_trunc = scrape_tpu_exporter(url, timeout_s=2.0)
        self.assertFalse(telem_trunc.reachable)
        self.assertEqual(telem_trunc.error, "exporter_incomplete_response")
        self.assertFalse(evaluate_tpu_readiness(telem_trunc)[0])

        # 2. Duplicate matching Content-Length with exact body -> succeeds cleanly.
        mode[0] = "dup_exact"
        telem_ok = scrape_tpu_exporter(url, timeout_s=2.0)
        self.assertTrue(telem_ok.reachable)
        self.assertTrue(telem_ok.exporter_up)
        self.assertTrue(evaluate_tpu_readiness(telem_ok)[0])

        # 3. Conflicting Content-Length -> rejected.
        mode[0] = "conflicting"
        telem_conflict = scrape_tpu_exporter(url, timeout_s=2.0)
        self.assertFalse(telem_conflict.reachable)
        self.assertFalse(evaluate_tpu_readiness(telem_conflict)[0])

        # 4. Malformed Content-Length -> rejected.
        mode[0] = "malformed"
        telem_bad = scrape_tpu_exporter(url, timeout_s=2.0)
        self.assertFalse(telem_bad.reachable)
        self.assertFalse(evaluate_tpu_readiness(telem_bad)[0])

    def test_completions_echo_token_id_prompt_excludes_echoed_tokens_from_fallback(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        class _EchoTokenIdsPromptHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return None

            def do_GET(self) -> None:  # noqa: N802
                body = json.dumps(
                    {"object": "list", "data": [{"id": "m-echo-tokid-prompt"}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                clen = int(self.headers.get("Content-Length", "0"))
                req_obj = json.loads(self.rfile.read(clen).decode("utf-8"))
                if req_obj.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    # Echoed 100-token prompt (~400 chars) + 12 chars of generated output, no usage.
                    ev1 = json.dumps(
                        {
                            "id": "cmpl-tokid-s1",
                            "object": "text_completion",
                            "choices": [{"index": 0, "text": "P" * 400, "finish_reason": None}],
                        }
                    ).encode("utf-8")
                    ev2 = json.dumps(
                        {
                            "id": "cmpl-tokid-s1",
                            "object": "text_completion",
                            "choices": [{"index": 0, "text": "Short answer", "finish_reason": "stop"}],
                        }
                    ).encode("utf-8")
                    self.wfile.write(
                        b"data: " + ev1 + b"\n\ndata: " + ev2 + b"\n\ndata: [DONE]\n\n"
                    )
                    self.wfile.flush()
                else:
                    body = json.dumps(
                        {
                            "id": "cmpl-tokid-ns1",
                            "object": "text_completion",
                            "choices": [
                                {"index": 0, "text": ("P" * 400) + "Short answer", "finish_reason": "stop"},
                                {"index": 1, "text": ("Q" * 200) + "Eight ch", "finish_reason": "stop"},
                            ],
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)

        srv = self._track(
            ThreadingHTTPServer(("127.0.0.1", 0), _EchoTokenIdsPromptHandler)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="echo-tokid-b",
                    type="pixel-tpu",
                    local=True,
                    base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    models=["m-echo-tokid-prompt"],
                )
            ],
        )
        router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
        rport = router.server_address[1]

        # 1. Streamed single token-ID prompt (100 tokens) with echo=True: 412 chars -> 103 - 100 = 3 tokens.
        st1, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo-tokid-prompt",
                "prompt": list(range(100)),
                "echo": True,
                "stream": True,
            },
        )
        self.assertEqual(st1, 200)
        self.assertEqual(recorded_usage[0].prompt_tokens, 100)
        self.assertEqual(recorded_usage[0].completion_tokens, 3)

        # 2. Non-streamed batched token-ID prompts (100 + 50 tokens) with echo=True: (103-100) + (52-50) = 5 tokens.
        st2, _, _ = _http_request(
            rport,
            "POST",
            "/v1/completions",
            body={
                "model": "m-echo-tokid-prompt",
                "prompt": [list(range(100)), list(range(50))],
                "echo": True,
                "stream": False,
            },
        )
        self.assertEqual(st2, 200)
        self.assertEqual(recorded_usage[1].prompt_tokens, 150)
        self.assertEqual(recorded_usage[1].completion_tokens, 5)

    def test_upstream_http_500_fails_over_and_cools_down_failing_backend(self) -> None:
        failing = self._track(
            _MockOpenAIBackend("vllm-500-primary", ["m-500"], status_code=200)
        )
        backup = self._track(
            _MockOpenAIBackend("vllm-500-backup", ["m-500"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="vllm-500-primary",
                    local=True,
                    base_url=failing.base_url,
                    models=["m-500"],
                ),
                BackendConfig(
                    id="vllm-500-backup",
                    local=False,
                    base_url=backup.base_url,
                    models=["m-500"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        failing.status_code = 500
        st, hdrs, _ = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-500",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(hdrs.get("x-freetpu-backend"), "vllm-500-backup")
        self.assertEqual(hdrs.get("x-freetpu-failovers"), "1")
        primary_state = router.engine._backends[0]
        self.assertEqual(primary_state.total_failures, 1)
        self.assertGreater(primary_state.cooldown_until_monotonic, time.monotonic())

    def test_high_batch_v6e8_preset_probes_vllm_health_and_ipv6_banner_brackets_host(
        self,
    ) -> None:
        from router.freetpu_router import _format_listen_authority, main as router_main

        preset = load_config(
            REPO_ROOT / "router" / "examples" / "cloud-tpu-v6e8-high-batch.json"
        )
        self.assertEqual(len(preset.backends), 2)
        for b in preset.backends:
            self.assertEqual(b.health_path, "/health")
            self.assertTrue(b.discover_models)

        self.assertEqual(_format_listen_authority("::1", 8080), "[::1]:8080")
        self.assertEqual(_format_listen_authority("::", 8080), "[::]:8080")
        self.assertEqual(_format_listen_authority("127.0.0.1", 8080), "127.0.0.1:8080")

        fake_srv = mock.MagicMock()
        fake_srv.server_address = ("::1", 8080, 0, 0)
        fake_srv.serve_forever.side_effect = KeyboardInterrupt()
        out_buf = io.StringIO()
        with (
            mock.patch("router.freetpu_router.create_router_server", return_value=fake_srv),
            mock.patch("sys.stdout", out_buf),
        ):
            rc = router_main(
                [
                    "--config",
                    str(REPO_ROOT / "router" / "examples" / "local-pixel-cluster.json"),
                    "--host",
                    "::1",
                    "--no-initial-probe",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertIn("http://[::1]:8080/v1", out_buf.getvalue())

    def test_tpu_exporter_normalizes_chunked_transfer_encoding_with_trailing_ows(
        self,
    ) -> None:
        metrics_payload = (
            b"freetpu_exporter_up 1\n"
            b"freetpu_tpu_devices_discovered 1\n"
            b'freetpu_tpu_health_status{chip="0"} 1\n'
            b'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
        )
        chunked_wire = (
            f"{len(metrics_payload):x}\r\n".encode("ascii")
            + metrics_payload
            + b"\r\n0\r\n\r\n"
        )
        mode = ["chunked_ows"]
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        stop_evt = threading.Event()
        release_conn = threading.Event()

        def _serve() -> None:
            listener.settimeout(0.2)
            while not stop_evt.is_set():
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                try:
                    conn.settimeout(1.0)
                    conn.recv(4096)
                    if mode[0] == "chunked_ows":
                        # Trailing OWS in Transfer-Encoding + persistent connection (no Connection: close).
                        conn.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: text/plain; version=0.0.4\r\n"
                            b"Transfer-Encoding: chunked  \t\r\n"
                            b"\r\n"
                            + chunked_wire
                        )
                        # Keep socket open until the scraper returns on the terminal 0-chunk.
                        release_conn.wait(timeout=5.0)
                    else:
                        conn.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: text/plain; version=0.0.4\r\n"
                            b"Transfer-Encoding: gzip\r\n"
                            b"Connection: close\r\n"
                            b"\r\n"
                            + metrics_payload
                        )
                except OSError:
                    pass
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        try:
            telem = scrape_tpu_exporter(
                f"http://127.0.0.1:{port}/metrics",
                timeout_s=1.0,
            )
            release_conn.set()
            self.assertTrue(telem.reachable, msg=f"unexpected error: {telem.error}")
            self.assertTrue(telem.exporter_up)
            self.assertEqual(telem.healthy_chips, 1)
            self.assertAlmostEqual(telem.max_hbm_utilization_ratio or 0.0, 0.25)
            self.assertTrue(evaluate_tpu_readiness(telem)[0])

            mode[0] = "unsupported_te"
            telem_bad = scrape_tpu_exporter(
                f"http://127.0.0.1:{port}/metrics",
                timeout_s=1.0,
            )
            self.assertFalse(telem_bad.reachable)
            self.assertFalse(evaluate_tpu_readiness(telem_bad)[0])
        finally:
            release_conn.set()
            stop_evt.set()
            listener.close()
            thread.join(timeout=2.0)

    def test_windows_selector_cap_and_reactor_value_error_releases_waiting_tasks(
        self,
    ) -> None:
        import selectors as selectors_mod
        from router import freetpu_router as router_mod

        with mock.patch.object(router_mod.sys, "platform", "win32"):
            self.assertEqual(
                router_mod._effective_max_concurrent_connections(256), 250
            )
            self.assertEqual(
                router_mod._effective_max_concurrent_connections(100), 100
            )
        with mock.patch.object(
            router_mod.selectors, "DefaultSelector", selectors_mod.SelectSelector
        ):
            self.assertEqual(
                router_mod._effective_max_concurrent_connections(512), 250
            )

        # If selector.select() raises ValueError, reactor exits cleanly and releases waiting tasks.
        reactor = router_mod._SSEReactor()
        up_a, up_b = socket.socketpair()
        cli_a, cli_b = socket.socketpair()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="sel-err-b",
                        local=True,
                        base_url="http://127.0.0.1:8081/v1",
                        models=["m1"],
                    )
                ],
            )
            engine = router_mod.RouterEngine(cfg)
            task = router_mod._SSEReactorTask(
                upstream_sock=up_a,
                client_sock=cli_a,
                engine=engine,
                sse_buf=bytearray(),
                raw_buf=bytearray(),
                chunked=False,
                chunk_left=None,
                remaining_content_length=None,
                max_event_bytes=65536,
                timeout_s=5.0,
                progress_deadline=time.monotonic() + 5.0,
                sse_mem_box=[0],
                on_event_fn=lambda _t, _l: (False, False),
                event_data_lines=[],
                current_event_type="",
                chunked_downstream=False,
            )
            with mock.patch.object(
                reactor._selector,
                "select",
                side_effect=ValueError("too many file descriptors in select()"),
            ):
                reactor.submit(task)
                self.assertTrue(
                    task.done_event.wait(timeout=2.0),
                    "Waiting handler deadlocked after selector ValueError",
                )
            self.assertTrue(task.upstream_read_error)
        finally:
            reactor.stop()
            for s in (up_a, up_b, cli_a, cli_b):
                try:
                    s.close()
                except OSError:
                    pass

    def test_upstream_http_429_fails_over_and_cools_down_overloaded_backend(
        self,
    ) -> None:
        overloaded = self._track(
            _MockOpenAIBackend("tgi-429-primary", ["m-429"], status_code=200)
        )
        backup = self._track(
            _MockOpenAIBackend("tgi-429-backup", ["m-429"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            backends=[
                BackendConfig(
                    id="tgi-429-primary",
                    local=True,
                    base_url=overloaded.base_url,
                    models=["m-429"],
                ),
                BackendConfig(
                    id="tgi-429-backup",
                    local=False,
                    base_url=backup.base_url,
                    models=["m-429"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        overloaded.status_code = 429
        st, hdrs, _ = _http_request(
            router.server_address[1],
            "POST",
            "/v1/chat/completions",
            body={
                "model": "m-429",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(hdrs.get("x-freetpu-backend"), "tgi-429-backup")
        self.assertEqual(hdrs.get("x-freetpu-failovers"), "1")
        primary_state = router.engine._backends[0]
        self.assertEqual(primary_state.total_failures, 1)
        self.assertGreater(primary_state.cooldown_until_monotonic, time.monotonic())

    def test_tpu_readiness_requires_duty_cycle_when_limit_configured_and_rejects_negative_hbm_ratio(
        self,
    ) -> None:
        # 1. Missing duty-cycle sample fails readiness when max_duty_cycle_percent is configured.
        no_duty_metrics = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 1\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
        )
        exp_no_duty = self._track(_MockTpuExporter(raw_body=no_duty_metrics))
        telem_no_duty = scrape_tpu_exporter(exp_no_duty.metrics_url, timeout_s=2.0)
        self.assertTrue(telem_no_duty.reachable)
        self.assertTrue(telem_no_duty.exporter_up)
        self.assertIsNone(telem_no_duty.max_duty_cycle_percent)
        self.assertTrue(evaluate_tpu_readiness(telem_no_duty)[0])
        ready_duty, reason_duty = evaluate_tpu_readiness(
            telem_no_duty, max_duty_cycle_percent=90.0
        )
        self.assertFalse(ready_duty)
        self.assertEqual(reason_duty, "tpu_duty_cycle_missing")

        # 2. Negative HBM ratio (-1) alongside 98% byte counters is rejected as invalid telemetry.
        neg_ratio_metrics = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 1\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_hbm_used_bytes{chip="0"} 9800000000\n'
            'freetpu_tpu_hbm_total_bytes{chip="0"} 10000000000\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} -1\n'
        )
        exp_neg = self._track(_MockTpuExporter(raw_body=neg_ratio_metrics))
        telem_neg = scrape_tpu_exporter(exp_neg.metrics_url, timeout_s=2.0)
        self.assertTrue(telem_neg.reachable)
        self.assertFalse(telem_neg.exporter_up)
        self.assertIn("freetpu_tpu_hbm_utilization_ratio", telem_neg.error or "")
        ready_neg, reason_neg = evaluate_tpu_readiness(telem_neg, max_hbm_utilization=0.95)
        self.assertFalse(ready_neg)
        self.assertEqual(reason_neg, "tpu_exporter_invalid_metrics")

    def test_probe_runs_tpu_readiness_before_stalled_optional_model_discovery(
        self,
    ) -> None:
        release_models = threading.Event()
        events_order: List[str] = []

        class _SlowModelsHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    events_order.append("health")
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                if self.path == "/v1/models":
                    events_order.append("models")
                    release_models.wait(timeout=2.0)
                    body = b'{"object":"list","data":[{"id":"m-probe"}]}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass
                    return
                self.send_response(404)
                self.end_headers()

        backend_srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowModelsHandler)
        backend_thread = threading.Thread(target=backend_srv.serve_forever, daemon=True)
        backend_thread.start()

        class _RecordingExporterHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                events_order.append("tpu_exporter")
                body = (
                    b"freetpu_exporter_up 1\n"
                    b"freetpu_tpu_devices_discovered 1\n"
                    b'freetpu_tpu_health_status{chip="0"} 1\n'
                    b'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.2\n'
                    b'freetpu_tpu_duty_cycle_percent{chip="0"} 15\n'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        exp_srv = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingExporterHandler)
        exp_thread = threading.Thread(target=exp_srv.serve_forever, daemon=True)
        exp_thread.start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                probe_timeout_s=0.35,
                backends=[
                    BackendConfig(
                        id="tpu-slow-models",
                        base_url=f"http://127.0.0.1:{backend_srv.server_address[1]}",
                        models=["m-probe"],
                        health_path="/health",
                        discover_models=True,
                        tpu_exporter_url=f"http://127.0.0.1:{exp_srv.server_address[1]}/metrics",
                        max_duty_cycle_percent=90.0,
                    )
                ],
            )
            router = self._track(create_router_server(cfg))
            state = router.engine._backends[0]
            self.assertTrue(state.healthy, f"Expected healthy worker, got {state.unhealthy_reason}")
            self.assertIsNotNone(state.tpu_telemetry)
            assert state.tpu_telemetry is not None
            self.assertTrue(state.tpu_telemetry.reachable)
            self.assertGreaterEqual(len(events_order), 2)
            self.assertEqual(events_order[0], "health")
            self.assertEqual(events_order[1], "tpu_exporter")
        finally:
            release_models.set()
            backend_srv.shutdown()
            backend_srv.server_close()
            exp_srv.shutdown()
            exp_srv.server_close()

    def test_streaming_chat_echo_prefix_buffers_before_commit_and_fails_over_on_pre_token_crash(
        self,
    ) -> None:
        class _EchoThenCrashHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-chat-echo"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Emit role chunk + echoed assistant prefix ("Prefix") then drop before generating.
                self.wfile.write(
                    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                    b'data: {"choices":[{"index":0,"delta":{"content":"Prefix"}}]}\n\n'
                )
                self.wfile.flush()
                self.close_connection = True

        primary_srv = ThreadingHTTPServer(("127.0.0.1", 0), _EchoThenCrashHandler)
        threading.Thread(target=primary_srv.serve_forever, daemon=True).start()

        class _EchoThenGenerateHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-chat-echo"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                body = (
                    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                    b'data: {"choices":[{"index":0,"delta":{"content":"Prefix"}}]}\n\n'
                    b'data: {"choices":[{"index":0,"delta":{"content":" generated"},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        backup_srv = ThreadingHTTPServer(("127.0.0.1", 0), _EchoThenGenerateHandler)
        threading.Thread(target=backup_srv.serve_forever, daemon=True).start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                backends=[
                    BackendConfig(
                        id="echo-crash-primary",
                        local=True,
                        base_url=f"http://127.0.0.1:{primary_srv.server_address[1]}",
                        models=["m-chat-echo"],
                    ),
                    BackendConfig(
                        id="echo-ok-backup",
                        local=False,
                        base_url=f"http://127.0.0.1:{backup_srv.server_address[1]}",
                        models=["m-chat-echo"],
                    ),
                ],
            )
            router = self._track(create_router_server(cfg))
            st, hdrs, bd = _http_request(
                router.server_address[1],
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-chat-echo",
                    "stream": True,
                    "echo": True,
                    "messages": [
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "Prefix"},
                    ],
                },
            )
            self.assertEqual(st, 200)
            self.assertEqual(hdrs.get("x-freetpu-backend"), "echo-ok-backup")
            self.assertEqual(hdrs.get("x-freetpu-failovers"), "1")
            self.assertIn(b"Prefix", bd)
            self.assertIn(b" generated", bd)
            self.assertIn(b"[DONE]", bd)
        finally:
            primary_srv.shutdown()
            primary_srv.server_close()
            backup_srv.shutdown()
            backup_srv.server_close()

    def test_chunked_post_inference_body_succeeds_and_enforces_size_and_conflict_limits(
        self,
    ) -> None:
        backend = self._track(_MockOpenAIBackend("chunked-backend", ["m-chunked"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_request_body_bytes=256,
            backends=[
                BackendConfig(
                    id="chunked-backend",
                    base_url=backend.base_url,
                    models=["m-chunked"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]

        def _send_raw(raw_http: bytes) -> Tuple[int, bytes]:
            with socket.create_connection(("127.0.0.1", rport), timeout=5.0) as sock:
                sock.sendall(raw_http)
                resp = bytearray()
                while True:
                    part = sock.recv(4096)
                    if not part:
                        break
                    resp.extend(part)
            head, _, body = bytes(resp).partition(b"\r\n\r\n")
            status_code = int(head.split(b"\r\n", 1)[0].split(b" ")[1])
            return status_code, body

        # 1. Valid multi-chunk POST /v1/chat/completions with chunk extension and trailer succeeds.
        payload = json.dumps(
            {
                "model": "m-chunked",
                "messages": [{"role": "user", "content": "hello chunked"}],
            }
        ).encode("utf-8")
        c1, c2 = payload[:17], payload[17:]
        chunked_req = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            + f"{len(c1):x};ext=1\r\n".encode("ascii")
            + c1
            + b"\r\n"
            + f"{len(c2):x}\r\n".encode("ascii")
            + c2
            + b"\r\n0\r\nX-Trailer: ok\r\n\r\n"
        )
        st_ok, body_ok = _send_raw(chunked_req)
        self.assertEqual(st_ok, 200)
        self.assertIn(b"chunked-backend", body_ok)

        # 2. Oversized chunked body exceeding max_request_body_bytes returns 413.
        big_payload = b'{"model":"m-chunked","messages":[{"role":"user","content":"' + (b"x" * 300) + b'"}]}'
        oversized_req = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            + f"{len(big_payload):x}\r\n".encode("ascii")
            + big_payload
            + b"\r\n0\r\n\r\n"
        )
        st_413, _ = _send_raw(oversized_req)
        self.assertEqual(st_413, 413)

        # 3. Conflicting Transfer-Encoding + Content-Length returns 400 per RFC 9112 §6.3.
        conflict_req = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Content-Length: 5\r\n"
            b"\r\n0\r\n\r\n"
        )
        st_400, _ = _send_raw(conflict_req)
        self.assertEqual(st_400, 400)

    def test_tpu_readiness_requires_duty_cycle_on_all_chips_when_limit_configured(
        self,
    ) -> None:
        partial_duty_metrics = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 2\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_health_status{chip="1"} 1\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="1"} 0.30\n'
            'freetpu_tpu_duty_cycle_percent{chip="0"} 20\n'
        )
        exp_partial = self._track(_MockTpuExporter(raw_body=partial_duty_metrics))
        telem_partial = scrape_tpu_exporter(exp_partial.metrics_url, timeout_s=2.0)
        self.assertEqual(telem_partial.total_chips, 2)
        self.assertEqual(telem_partial.chips_with_duty_cycle, 1)
        self.assertEqual(telem_partial.max_duty_cycle_percent, 20.0)
        self.assertTrue(evaluate_tpu_readiness(telem_partial)[0])
        ready_partial, reason_partial = evaluate_tpu_readiness(
            telem_partial, max_duty_cycle_percent=90.0
        )
        self.assertFalse(ready_partial)
        self.assertEqual(reason_partial, "tpu_duty_cycle_missing")

        full_duty_metrics = partial_duty_metrics + 'freetpu_tpu_duty_cycle_percent{chip="1"} 40\n'
        exp_full = self._track(_MockTpuExporter(raw_body=full_duty_metrics))
        telem_full = scrape_tpu_exporter(exp_full.metrics_url, timeout_s=2.0)
        self.assertEqual(telem_full.chips_with_duty_cycle, 2)
        ready_full, reason_full = evaluate_tpu_readiness(
            telem_full, max_duty_cycle_percent=90.0
        )
        self.assertTrue(ready_full)
        self.assertIsNone(reason_full)

    def test_streaming_chat_echo_excludes_echoed_assistant_prefix_from_fallback_usage(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        echoed_prefix = "A" * 400  # 100 estimated tokens if wrongly counted
        generated_tail = "12345678"  # 8 chars -> 2 estimated completion tokens

        class _ChatEchoStreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-chat-echo-usage"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                chunk1 = json.dumps(
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
                ).encode("utf-8")
                chunk2 = json.dumps(
                    {"choices": [{"index": 0, "delta": {"content": echoed_prefix}}]}
                ).encode("utf-8")
                chunk3 = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": generated_tail},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode("utf-8")
                body = (
                    b"data: " + chunk1 + b"\n\n"
                    b"data: " + chunk2 + b"\n\n"
                    b"data: " + chunk3 + b"\n\n"
                    b"data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChatEchoStreamHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="chat-echo-usage",
                        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                        models=["m-chat-echo-usage"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
            st, _, bd = _http_request(
                router.server_address[1],
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-chat-echo-usage",
                    "stream": True,
                    "echo": True,
                    "messages": [
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": echoed_prefix},
                    ],
                },
            )
            self.assertEqual(st, 200)
            self.assertIn(generated_tail.encode("utf-8"), bd)
            self.assertEqual(len(recorded_usage), 1)
            self.assertEqual(recorded_usage[0].completion_tokens, 2)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_invalid_url_without_authority_redacts_credentials_in_diagnostics(
        self,
    ) -> None:
        bad_urls = (
            "http:alice:supersecret@host:8000",
            "alice:supersecret@host:8000",
            "http:///alice:supersecret@host:8000",
            "alice:supersecret@http://bad\u2100host/v1",
        )
        for bad_url in bad_urls:
            with self.subTest(bad_url=bad_url):
                sanitized = sanitize_url(bad_url)
                self.assertNotIn("alice", sanitized)
                self.assertNotIn("supersecret", sanitized)
                with self.assertRaises(ValueError) as ctx:
                    validate_http_url(bad_url, field_name="backend base_url", allow_query=False)
                err_msg = str(ctx.exception)
                self.assertNotIn("alice", err_msg)
                self.assertNotIn("supersecret", err_msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "bad_url.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "backends": [
                            {
                                "id": "bad-cred-url",
                                "base_url": "http:alice:supersecret@host:8000",
                                "models": ["m1"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc = router_main(["--config", str(cfg_path), "--check-config"])
            self.assertNotEqual(rc, 0)
            stderr_out = err_buf.getvalue()
            self.assertNotIn("alice", stderr_out)
            self.assertNotIn("supersecret", stderr_out)

    def test_tpu_readiness_requires_hbm_coverage_across_all_chips_when_hbm_reported(
        self,
    ) -> None:
        partial_hbm_metrics = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 2\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_health_status{chip="1"} 1\n'
            'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.25\n'
        )
        exp_partial = self._track(_MockTpuExporter(raw_body=partial_hbm_metrics))
        telem_partial = scrape_tpu_exporter(exp_partial.metrics_url, timeout_s=2.0)
        self.assertEqual(telem_partial.total_chips, 2)
        self.assertEqual(telem_partial.chips_with_hbm, 1)
        ready_p, reason_p = evaluate_tpu_readiness(
            telem_partial, max_hbm_utilization=0.95
        )
        self.assertFalse(ready_p)
        self.assertEqual(reason_p, "tpu_hbm_missing")

        full_hbm_metrics = (
            partial_hbm_metrics
            + 'freetpu_tpu_hbm_used_bytes{chip="1"} 4000000000\n'
            + 'freetpu_tpu_hbm_total_bytes{chip="1"} 16000000000\n'
        )
        exp_full = self._track(_MockTpuExporter(raw_body=full_hbm_metrics))
        telem_full = scrape_tpu_exporter(exp_full.metrics_url, timeout_s=2.0)
        self.assertEqual(telem_full.chips_with_hbm, 2)
        ready_f, reason_f = evaluate_tpu_readiness(telem_full, max_hbm_utilization=0.95)
        self.assertTrue(ready_f)
        self.assertIsNone(reason_f)

    def test_streaming_ttft_measures_first_generated_output_arrival_and_skips_echo_and_finish_only(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        mode = ["finish_only"]

        class _TTFTStreamHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-ttft"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode[0] == "finish_only":
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                        b"data: [DONE]\n\n"
                    )
                    self.wfile.flush()
                elif mode[0] == "echo_only":
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                        b'data: {"choices":[{"index":0,"delta":{"content":"Prefix"}}]}\n\n'
                        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
                        b"data: [DONE]\n\n"
                    )
                    self.wfile.flush()
                elif mode[0] == "echo_then_delayed_gen":
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
                        b'data: {"choices":[{"index":0,"delta":{"content":"Prefix"}}]}\n\n'
                    )
                    self.wfile.flush()
                    time.sleep(0.15)
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"content":" generated"},"finish_reason":"stop"}]}\n\n'
                        b"data: [DONE]\n\n"
                    )
                    self.wfile.flush()
                else:  # prompt_progress_then_gen
                    for i in range(5):
                        self.wfile.write(
                            f'data: {{"prompt_progress":{{"processed":{i + 1},"total":5}}}}\n\n'.encode(
                                "utf-8"
                            )
                        )
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
                        b"data: [DONE]\n\n"
                    )
                    self.wfile.flush()
                self.close_connection = True

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _TTFTStreamHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                backends=[
                    BackendConfig(
                        id="ttft-backend",
                        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                        models=["m-ttft"],
                    )
                ],
            )
            router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
            rport = router.server_address[1]

            # 1. Finish-only stream produces no generated token -> ttft_ms must be None and
            #    no TTFT histogram sample is recorded.
            st1, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-ttft",
                    "stream": True,
                    "messages": [{"role": "user", "content": "empty"}],
                },
            )
            self.assertEqual(st1, 200)
            self.assertEqual(len(recorded_usage), 1)
            self.assertIsNone(recorded_usage[0].ttft_ms)

            # 2. Echo-only stream (echo=true, finish_reason=length without generated output)
            #    must also leave ttft_ms as None.
            mode[0] = "echo_only"
            st2, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-ttft",
                    "stream": True,
                    "echo": True,
                    "messages": [
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "Prefix"},
                    ],
                },
            )
            self.assertEqual(st2, 200)
            self.assertEqual(len(recorded_usage), 2)
            self.assertIsNone(recorded_usage[1].ttft_ms)
            self.assertNotIn("freetpu_router_ttft_seconds_count", router.engine.metrics_text())

            # 3. Echoed prefix followed 150ms later by generated token: TTFT waits for the
            #    generated token (>= 120ms), NOT the immediate echoed prefix.
            mode[0] = "echo_then_delayed_gen"
            st3, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-ttft",
                    "stream": True,
                    "echo": True,
                    "messages": [
                        {"role": "user", "content": "continue"},
                        {"role": "assistant", "content": "Prefix"},
                    ],
                },
            )
            self.assertEqual(st3, 200)
            self.assertEqual(len(recorded_usage), 3)
            self.assertIsNotNone(recorded_usage[2].ttft_ms)
            assert recorded_usage[2].ttft_ms is not None
            self.assertGreaterEqual(recorded_usage[2].ttft_ms, 120.0)
            self.assertIn(
                'freetpu_router_ttft_seconds_count{model="m-ttft",backend_id="ttft-backend"} 1',
                router.engine.metrics_text(),
            )

            # 4. Phase 1 arrival timestamp is preserved even if header emission / replay is slow.
            mode[0] = "prompt_progress_then_gen"
            import router.freetpu_router as router_mod

            orig_end_headers = router_mod._RouterRequestHandler.end_headers

            def _slow_end_headers(handler_self: Any) -> None:
                if getattr(handler_self, "path", "") == "/v1/chat/completions":
                    time.sleep(0.15)
                orig_end_headers(handler_self)

            with mock.patch.object(
                router_mod._RouterRequestHandler, "end_headers", _slow_end_headers
            ):
                st4, _, _ = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-ttft",
                        "stream": True,
                        "messages": [{"role": "user", "content": "fast"}],
                    },
                )
            self.assertEqual(st4, 200)
            self.assertEqual(len(recorded_usage), 4)
            ev4 = recorded_usage[3]
            self.assertIsNotNone(ev4.ttft_ms)
            assert ev4.ttft_ms is not None
            self.assertGreaterEqual(ev4.latency_ms, 140.0)
            # TTFT was captured at Phase 1 arrival BEFORE the 150ms header/replay stall.
            self.assertLess(ev4.ttft_ms, ev4.latency_ms - 100.0)
            self.assertIn(
                'freetpu_router_ttft_seconds_count{model="m-ttft",backend_id="ttft-backend"} 2',
                router.engine.metrics_text(),
            )
        finally:
            srv.shutdown()
            srv.server_close()

    def test_upstream_http_401_fails_over_to_healthy_secondary_while_client_401_stays_immediate(
        self,
    ) -> None:
        bad_key_primary = self._track(
            _MockOpenAIBackend("auth-401-primary", ["m-401"], status_code=200)
        )
        good_backup = self._track(
            _MockOpenAIBackend("auth-401-backup", ["m-401"], status_code=200)
        )
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            policy="local_first",
            failure_cooldown_s=30.0,
            auth={"type": "bearer", "tokens": ["router-client-secret"]},
            backends=[
                BackendConfig(
                    id="auth-401-primary",
                    local=True,
                    base_url=bad_key_primary.base_url,
                    models=["m-401"],
                ),
                BackendConfig(
                    id="auth-401-backup",
                    local=False,
                    base_url=good_backup.base_url,
                    models=["m-401"],
                ),
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        bad_key_primary.status_code = 401

        # 1. Invalid client bearer token still gets immediate 401 from router without touching backends.
        st_unauth, _, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-401", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-client-token"},
        )
        self.assertEqual(st_unauth, 401)
        self.assertEqual(len(good_backup.requests_seen), 0)

        # 2. Valid client token: upstream 401 from primary fails over to healthy secondary.
        st_ok, hdrs_ok, _ = _http_request(
            rport,
            "POST",
            "/v1/chat/completions",
            body={"model": "m-401", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer router-client-secret"},
        )
        self.assertEqual(st_ok, 200)
        self.assertEqual(hdrs_ok.get("x-freetpu-backend"), "auth-401-backup")
        self.assertEqual(hdrs_ok.get("x-freetpu-failovers"), "1")
        self.assertEqual(router.engine._backends[0].total_failures, 1)

    def test_no_initial_probe_holds_exporter_gated_backend_unready_until_first_scrape(
        self,
    ) -> None:
        allow_exporter = threading.Event()
        exporter_calls = [0]

        class _GatedExporterHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                exporter_calls[0] += 1
                allow_exporter.wait(timeout=3.0)
                body = (
                    b"freetpu_exporter_up 1\n"
                    b"freetpu_tpu_devices_discovered 1\n"
                    b'freetpu_tpu_health_status{chip="0"} 1\n'
                    b'freetpu_tpu_hbm_utilization_ratio{chip="0"} 0.2\n'
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        exp_srv = ThreadingHTTPServer(("127.0.0.1", 0), _GatedExporterHandler)
        threading.Thread(target=exp_srv.serve_forever, daemon=True).start()
        tpu_backend = self._track(_MockOpenAIBackend("tpu-async-start", ["m-async"]))
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                health_poll_interval_s=60.0,
                probe_timeout_s=2.0,
                backends=[
                    BackendConfig(
                        id="tpu-async-start",
                        base_url=tpu_backend.base_url,
                        models=["m-async"],
                        tpu_exporter_url=f"http://127.0.0.1:{exp_srv.server_address[1]}/metrics",
                    )
                ],
            )
            router = self._track(create_router_server(cfg, probe_on_start=False))
            rport = router.server_address[1]

            # Before exporter readiness completes, /health is 503 and inference must not route to it.
            st_h, _, bd_h = _http_request(rport, "GET", "/health")
            self.assertEqual(st_h, 503)
            self.assertIn(b"tpu_readiness_pending", bd_h)
            st_inf, _, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-async",
                    "messages": [{"role": "user", "content": "too early"}],
                },
            )
            self.assertEqual(st_inf, 503)
            self.assertEqual(len(tpu_backend.requests_seen), 0)

            # Once exporter responds, immediate async startup poll marks backend ready without waiting 60s.
            allow_exporter.set()
            deadline = time.monotonic() + 2.0
            became_ready = False
            while time.monotonic() < deadline:
                st_now, _, _ = _http_request(rport, "GET", "/health")
                if st_now == 200:
                    became_ready = True
                    break
                time.sleep(0.05)
            self.assertTrue(
                became_ready,
                "Async startup poll did not mark exporter-gated backend ready promptly",
            )
            self.assertGreaterEqual(exporter_calls[0], 1)

            st_ok, hdrs_ok, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-async",
                    "messages": [{"role": "user", "content": "now ready"}],
                },
            )
            self.assertEqual(st_ok, 200)
            self.assertEqual(hdrs_ok.get("x-freetpu-backend"), "tpu-async-start")
        finally:
            allow_exporter.set()
            exp_srv.shutdown()
            exp_srv.server_close()

    def test_server_restart_restores_sse_reactor_streaming(self) -> None:
        backend = self._track(_MockOpenAIBackend("restart-sse-b", ["m-restart"]))
        backend.custom_sse_chunks = [
            {"choices": [{"index": 0, "delta": {"content": "chunk-one "}}]},
            {"choices": [{"index": 0, "delta": {"content": "chunk-two"}, "finish_reason": "stop"}]},
        ]
        backend.stream_chunk_delay_s = 0.02
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            backends=[
                BackendConfig(
                    id="restart-sse-b",
                    base_url=backend.base_url,
                    models=["m-restart"],
                )
            ],
        )
        router = create_router_server(cfg)
        rport = router.server_address[1]
        t1 = threading.Thread(target=router.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t1.start()
        try:
            st1, _, bd1 = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-restart",
                    "stream": True,
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
            self.assertEqual(st1, 200)
            self.assertIn(b"chunk-one ", bd1)
            self.assertIn(b"chunk-two", bd1)
            self.assertIn(b"[DONE]", bd1)

            # Shut down the server (stops SSE reactor), then restart serve_forever() on the same instance.
            router.shutdown()
            t1.join(timeout=2.0)
            self.assertTrue(router.sse_reactor.is_closed)

            t2 = threading.Thread(target=router.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            t2.start()
            try:
                st2, _, bd2 = _http_request(
                    rport,
                    "POST",
                    "/v1/chat/completions",
                    body={
                        "model": "m-restart",
                        "stream": True,
                        "messages": [{"role": "user", "content": "second"}],
                    },
                )
                self.assertEqual(st2, 200)
                self.assertIn(b"chunk-one ", bd2)
                self.assertIn(b"chunk-two", bd2)
                self.assertIn(b"[DONE]", bd2)
                self.assertFalse(router.sse_reactor.is_closed)
            finally:
                router.shutdown()
                t2.join(timeout=2.0)
        finally:
            router.server_close()

    def test_cors_allowed_origins_rejects_paths_trailing_slash_queries_and_credentials(
        self,
    ) -> None:
        base_backend = BackendConfig(
            id="cors-check-b",
            base_url="http://127.0.0.1:8000",
            models=["m1"],
        )
        ok_cfg = RouterConfig(
            host="127.0.0.1",
            port=8080,
            cors_allowed_origins=["https://app.example", "http://localhost:3000", "*"],
            backends=[base_backend],
        )
        ok_cfg.validate()
        self.assertEqual(
            ok_cfg.cors_allowed_origins,
            ["https://app.example", "http://localhost:3000", "*"],
        )

        invalid_entries = (
            "https://app.example/chat",
            "https://app.example/",
            "https://app.example?x=1",
            "https://app.example#frag",
            "https://user:secret@app.example",
            "ftp://app.example",
        )
        for bad_origin in invalid_entries:
            with self.subTest(bad_origin=bad_origin):
                with self.assertRaises(ValueError) as ctx:
                    RouterConfig(
                        host="127.0.0.1",
                        port=8080,
                        cors_allowed_origins=[bad_origin],
                        backends=[base_backend],
                    )
                self.assertNotIn("secret", str(ctx.exception))

    def test_legacy_completions_stream_echo_buffers_prompt_for_failover_and_skips_echo_only_ttft(
        self,
    ) -> None:
        recorded_usage: List[UsageEvent] = []

        class _RecordHook:
            def record_usage(self, event: UsageEvent) -> None:
                recorded_usage.append(event)

        prompt_str = "Legacy echo prompt"

        class _EchoDropHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-legacy-echo"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Emit only the echoed prompt and then drop connection before generating.
                echo_ev = json.dumps(
                    {"choices": [{"index": 0, "text": prompt_str}]}
                ).encode("utf-8")
                self.wfile.write(b"data: " + echo_ev + b"\n\n")
                self.wfile.flush()
                self.close_connection = True

        backup_mode = ["generate"]

        class _EchoOkHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def do_GET(self) -> None:  # noqa: N802
                body = b'{"object":"list","data":[{"id":"m-legacy-echo"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", "0"))
                if cl > 0:
                    self.rfile.read(cl)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if backup_mode[0] == "echo_only":
                    ev = json.dumps(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "text": prompt_str,
                                    "finish_reason": "length",
                                }
                            ]
                        }
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + ev + b"\n\ndata: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    ev1 = json.dumps(
                        {"choices": [{"index": 0, "text": prompt_str}]}
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + ev1 + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.12)
                    ev2 = json.dumps(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "text": " answer!",
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ).encode("utf-8")
                    self.wfile.write(b"data: " + ev2 + b"\n\ndata: [DONE]\n\n")
                    self.wfile.flush()
                self.close_connection = True

        primary_srv = ThreadingHTTPServer(("127.0.0.1", 0), _EchoDropHandler)
        backup_srv = ThreadingHTTPServer(("127.0.0.1", 0), _EchoOkHandler)
        threading.Thread(target=primary_srv.serve_forever, daemon=True).start()
        threading.Thread(target=backup_srv.serve_forever, daemon=True).start()
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                failure_cooldown_s=30.0,
                backends=[
                    BackendConfig(
                        id="legacy-echo-primary",
                        local=True,
                        base_url=f"http://127.0.0.1:{primary_srv.server_address[1]}",
                        models=["m-legacy-echo"],
                    ),
                    BackendConfig(
                        id="legacy-echo-backup",
                        local=False,
                        base_url=f"http://127.0.0.1:{backup_srv.server_address[1]}",
                        models=["m-legacy-echo"],
                    ),
                ],
            )
            router = self._track(create_router_server(cfg, usage_hook=_RecordHook()))
            rport = router.server_address[1]

            # 1. Primary drops after sending only echoed prompt -> router fails over to backup,
            #    and TTFT reflects delayed generation on backup (>= 100ms), not immediate echo.
            st1, hdrs1, bd1 = _http_request(
                rport,
                "POST",
                "/v1/completions",
                body={
                    "model": "m-legacy-echo",
                    "prompt": prompt_str,
                    "echo": True,
                    "stream": True,
                },
            )
            self.assertEqual(st1, 200)
            self.assertEqual(hdrs1.get("x-freetpu-backend"), "legacy-echo-backup")
            self.assertEqual(hdrs1.get("x-freetpu-failovers"), "1")
            self.assertIn(prompt_str.encode("utf-8"), bd1)
            self.assertIn(b" answer!", bd1)
            self.assertEqual(len(recorded_usage), 1)
            self.assertIsNotNone(recorded_usage[0].ttft_ms)
            assert recorded_usage[0].ttft_ms is not None
            self.assertGreaterEqual(recorded_usage[0].ttft_ms, 100.0)
            self.assertIn(
                'freetpu_router_ttft_seconds_count{model="m-legacy-echo",backend_id="legacy-echo-backup"} 1',
                router.engine.metrics_text(),
            )

            # 2. Echo-only max_tokens=0 stream commits cleanly with ttft_ms=None (count stays 1).
            backup_mode[0] = "echo_only"
            st2, _, bd2 = _http_request(
                rport,
                "POST",
                "/v1/completions",
                body={
                    "model": "m-legacy-echo",
                    "prompt": prompt_str,
                    "echo": True,
                    "max_tokens": 0,
                    "stream": True,
                },
            )
            self.assertEqual(st2, 200)
            self.assertIn(prompt_str.encode("utf-8"), bd2)
            self.assertEqual(len(recorded_usage), 2)
            self.assertIsNone(recorded_usage[1].ttft_ms)
            self.assertEqual(recorded_usage[1].completion_tokens, 0)
            self.assertIn(
                'freetpu_router_ttft_seconds_count{model="m-legacy-echo",backend_id="legacy-echo-backup"} 1',
                router.engine.metrics_text(),
            )
        finally:
            primary_srv.shutdown()
            primary_srv.server_close()
            backup_srv.shutdown()
            backup_srv.server_close()

    def test_tpu_readiness_fails_when_all_hbm_metrics_missing_unless_ceiling_opted_out(
        self,
    ) -> None:
        no_hbm_metrics = (
            "freetpu_exporter_up 1\n"
            "freetpu_tpu_devices_discovered 2\n"
            'freetpu_tpu_health_status{chip="0"} 1\n'
            'freetpu_tpu_health_status{chip="1"} 1\n'
        )
        exp = self._track(_MockTpuExporter(raw_body=no_hbm_metrics))
        telem = scrape_tpu_exporter(exp.metrics_url, timeout_s=2.0)
        self.assertEqual(telem.total_chips, 2)
        self.assertEqual(telem.healthy_chips, 2)
        self.assertEqual(telem.chips_with_hbm, 0)

        # With default / configured HBM ceiling (0.95), 0 HBM metrics must fail readiness.
        ready_default, reason_default = evaluate_tpu_readiness(
            telem, max_hbm_utilization=0.95
        )
        self.assertFalse(ready_default)
        self.assertEqual(reason_default, "tpu_hbm_missing")

        # Explicit opt-out (max_hbm_utilization=None) allows health-only readiness.
        ready_opt_out, reason_opt_out = evaluate_tpu_readiness(
            telem, max_hbm_utilization=None
        )
        self.assertTrue(ready_opt_out)
        self.assertIsNone(reason_opt_out)

        # Config validation also accepts explicit max_hbm_utilization=None opt-out.
        opt_out_backend = BackendConfig.from_dict(
            {
                "id": "tpu-no-hbm-limit",
                "base_url": "http://127.0.0.1:8000",
                "models": ["m1"],
                "max_hbm_utilization": None,
            }
        )
        self.assertIsNone(opt_out_backend.max_hbm_utilization)

    def test_malformed_urls_with_unescaped_password_delimiters_redact_all_credentials(
        self,
    ) -> None:
        invalid_secret_cases = (
            ("https://alice:p4/ss@host:8000", ("alice", "p4", "ss")),
            ("https://alice:p4?ss@host:8000", ("alice", "p4", "ss")),
            ("https://alice:p4#ss@host:8000", ("alice", "p4", "ss")),
            ("https://alice:99999/ss@host:8000", ("alice", "99999", "ss")),
        )
        for bad_url, forbidden_tokens in invalid_secret_cases:
            with self.subTest(bad_url=bad_url):
                sanitized = sanitize_url(bad_url)
                self.assertEqual(sanitized, "<redacted invalid URL>")
                for tok in forbidden_tokens:
                    self.assertNotIn(tok, sanitized)
                with self.assertRaises(ValueError) as ctx:
                    validate_http_url(bad_url, field_name="backend base_url", allow_query=False)
                err_msg = str(ctx.exception)
                for tok in forbidden_tokens:
                    self.assertNotIn(tok, err_msg)

        # Even if numeric password looks like a valid port before '/' + '@', sanitize_url redacts it.
        ambig = sanitize_url("https://alice:1234/w0rd@host:8000")
        self.assertEqual(ambig, "<redacted invalid URL>")
        for tok in ("alice", "1234", "w0rd"):
            self.assertNotIn(tok, ambig)

        # Backslash in password parsed as userinfo still strips all credentials cleanly.
        bs_sanitized = sanitize_url("https://alice:p4\\w0rd@host:8000")
        for tok in ("alice", "p4", "w0rd"):
            self.assertNotIn(tok, bs_sanitized)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "bad_pw_slash.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "backends": [
                            {
                                "id": "pw-slash-b",
                                "base_url": "https://alice:p4/w0rd@host:8000",
                                "models": ["m1"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc = router_main(["--config", str(cfg_path), "--check-config"])
            self.assertNotEqual(rc, 0)
            stderr_out = err_buf.getvalue()
            for tok in ("alice", "p4/w0rd", "p4", "w0rd"):
                self.assertNotIn(tok, stderr_out)

    def test_upstream_attempt_deadline_bounds_connect_plus_stalled_upload_and_fails_over(
        self,
    ) -> None:
        from router import freetpu_router as router_mod

        # 1. Unit check on _send_http_request_with_deadline: slow connect + slow trickle
        #    upload must raise TimeoutError at the single shared deadline (never per-chunk).
        class _TrickleSock:
            def __init__(self) -> None:
                self.timeouts: List[Optional[float]] = []

            def setsockopt(self, *args: Any, **kwargs: Any) -> None:
                pass

            def settimeout(self, t: Optional[float]) -> None:
                self.timeouts.append(t)

            def sendall(self, data: bytes) -> None:
                pass

            def send(self, data: Any) -> int:
                time.sleep(0.06)
                return min(1024, len(data))

        class _SlowConnectConn(http.client.HTTPConnection):
            def __init__(self) -> None:
                super().__init__("127.0.0.1", 80, timeout=1.0)
                self._trickle = _TrickleSock()

            def connect(self) -> None:
                time.sleep(0.15)
                self.sock = self._trickle  # type: ignore[assignment]

        t0 = time.monotonic()
        conn = _SlowConnectConn()
        with self.assertRaises(TimeoutError):
            router_mod._send_http_request_with_deadline(
                conn,
                "POST",
                "/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                body=b"x" * 65536,
                deadline=t0 + 0.30,
                label="test-upload",
            )
        elapsed_u = time.monotonic() - t0
        self.assertLess(elapsed_u, 0.55)
        self.assertGreaterEqual(len(conn._trickle.timeouts), 3)
        # Subsequent send timeouts must strictly shrink as deadline approaches.
        self.assertLess(conn._trickle.timeouts[-1] or 99.0, 0.20)

        # 2. End-to-end router failover when primary stalls during upload/response.
        hold_open = threading.Event()
        accepted_conns: List[socket.socket] = []
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        stalled_port = listener.getsockname()[1]

        def _accept_and_stall() -> None:
            listener.settimeout(0.2)
            while not hold_open.is_set():
                try:
                    c, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                try:
                    c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                except OSError:
                    pass
                accepted_conns.append(c)

        accept_thread = threading.Thread(target=_accept_and_stall, daemon=True)
        accept_thread.start()
        backup = self._track(_MockOpenAIBackend("deadline-backup", ["m-deadline"]))
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                request_timeout_s=0.30,
                max_request_body_bytes=4 * 1024 * 1024,
                max_total_request_memory_bytes=32 * 1024 * 1024,
                backends=[
                    BackendConfig(
                        id="stalled-primary",
                        local=True,
                        base_url=f"http://127.0.0.1:{stalled_port}",
                        models=["m-deadline"],
                        timeout_s=0.30,
                    ),
                    BackendConfig(
                        id="deadline-backup",
                        local=False,
                        base_url=backup.base_url,
                        models=["m-deadline"],
                    ),
                ],
            )
            router = self._track(create_router_server(cfg, probe_on_start=False))
            rport = router.server_address[1]
            # Wait briefly for backup to be ready, keep primary healthy to force attempt on it first.
            router.engine._backends[0].healthy = True
            router.engine._backends[1].healthy = True

            t_req = time.monotonic()
            st, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "m-deadline",
                    "messages": [{"role": "user", "content": "A" * (512 * 1024)}],
                },
            )
            elapsed_req = time.monotonic() - t_req
            self.assertEqual(st, 200)
            self.assertEqual(hdrs.get("x-freetpu-backend"), "deadline-backup")
            self.assertEqual(hdrs.get("x-freetpu-failovers"), "1")
            self.assertLess(elapsed_req, 1.5)
        finally:
            hold_open.set()
            listener.close()
            for c in accepted_conns:
                try:
                    c.close()
                except OSError:
                    pass
            accept_thread.join(timeout=1.0)

    def test_oversized_request_line_returns_http_431_instead_of_eof(self) -> None:
        backend = self._track(_MockOpenAIBackend("reqline-431-b", ["m-431"]))
        cfg = RouterConfig(
            host="127.0.0.1",
            port=0,
            max_header_bytes=512,
            backends=[
                BackendConfig(
                    id="reqline-431-b",
                    base_url=backend.base_url,
                    models=["m-431"],
                )
            ],
        )
        router = self._track(create_router_server(cfg))
        rport = router.server_address[1]
        oversized_req = (
            b"GET /" + (b"a" * 2048) + b" HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        )
        with socket.create_connection(("127.0.0.1", rport), timeout=5.0) as sock:
            sock.sendall(oversized_req)
            resp = bytearray()
            while True:
                try:
                    chunk = sock.recv(4096)
                except ConnectionResetError:
                    break
                if not chunk:
                    break
                resp.extend(chunk)
        raw_resp = bytes(resp)
        self.assertTrue(
            raw_resp.startswith(b"HTTP/1.1 431 "),
            f"Expected HTTP/1.1 431 response, got: {raw_resp[:120]!r}",
        )
        _, _, body = raw_resp.partition(b"\r\n\r\n")
        parsed = json.loads(body.decode("utf-8"))
        self.assertEqual(parsed["error"]["code"], "request_header_fields_too_large")

    def test_model_not_found_404_tries_other_backends_without_cooling_down_worker(
        self,
    ) -> None:
        def _make_selective_server(
            backend_id: str,
            served_model: str,
            *,
            top_level_vllm_error: bool = False,
        ) -> ThreadingHTTPServer:
            class _SelectiveHandler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                    pass

                def do_GET(self) -> None:  # noqa: N802
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header("Content-Length", "2")
                        self.end_headers()
                        self.wfile.write(b"ok")
                        return
                    body = json.dumps(
                        {"object": "list", "data": [{"id": served_model}]}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_POST(self) -> None:  # noqa: N802
                    cl = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(cl).decode("utf-8"))
                    req_model = req.get("model")
                    if req_model != served_model:
                        if top_level_vllm_error:
                            err_obj: Dict[str, Any] = {
                                "object": "error",
                                "message": f"The model `{req_model}` does not exist.",
                                "type": "NotFoundError",
                                "param": "model",
                                "code": 404,
                            }
                        else:
                            err_obj = {
                                "error": {
                                    "message": f"The model `{req_model}` does not exist.",
                                    "type": "NotFoundError",
                                    "param": "model",
                                    "code": 404,
                                }
                            }
                        body = json.dumps(err_obj).encode("utf-8")
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    ok_body = json.dumps(
                        {
                            "id": f"chatcmpl-{backend_id}",
                            "object": "chat.completion",
                            "model": served_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"served-by:{backend_id}",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(ok_body)))
                    self.end_headers()
                    self.wfile.write(ok_body)

            srv = ThreadingHTTPServer(("127.0.0.1", 0), _SelectiveHandler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            return srv

        primary_srv = _make_selective_server(
            "wildcard-primary", "local-model", top_level_vllm_error=False
        )
        fallback_srv = _make_selective_server(
            "fallback-llama", "llama", top_level_vllm_error=True
        )
        try:
            cfg = RouterConfig(
                host="127.0.0.1",
                port=0,
                policy="local_first",
                failure_cooldown_s=60.0,
                backends=[
                    BackendConfig(
                        id="wildcard-primary",
                        local=True,
                        discover_models=False,
                        health_path="/health",
                        base_url=f"http://127.0.0.1:{primary_srv.server_address[1]}",
                        models=["*"],
                    ),
                    BackendConfig(
                        id="fallback-llama",
                        local=False,
                        discover_models=False,
                        health_path="/health",
                        base_url=f"http://127.0.0.1:{fallback_srv.server_address[1]}",
                        models=["*"],
                    ),
                ],
            )
            router = self._track(create_router_server(cfg))
            rport = router.server_address[1]

            # (a) Request for `llama` tries wildcard primary (404 param=model), fails over to fallback.
            st1, hdrs1, bd1 = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "llama",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(st1, 200)
            self.assertEqual(hdrs1.get("x-freetpu-backend"), "fallback-llama")
            self.assertEqual(hdrs1.get("x-freetpu-failovers"), "1")
            self.assertIn(b"served-by:fallback-llama", bd1)

            # (b) Primary stays healthy (not cooled down) and serves its own model immediately.
            primary_state = router.engine._backends[0]
            self.assertTrue(primary_state.healthy)
            self.assertEqual(primary_state.cooldown_until_monotonic, 0.0)
            self.assertEqual(primary_state.total_failures, 0)

            st2, hdrs2, bd2 = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "local-model",
                    "messages": [{"role": "user", "content": "hi local"}],
                },
            )
            self.assertEqual(st2, 200)
            self.assertEqual(hdrs2.get("x-freetpu-backend"), "wildcard-primary")
            self.assertEqual(hdrs2.get("x-freetpu-failovers"), "0")
            self.assertIn(b"served-by:wildcard-primary", bd2)

            # (c) When NO candidate serves the model, router returns 404 cleanly to the client.
            st3, _, bd3 = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={
                    "model": "unknown-model",
                    "messages": [{"role": "user", "content": "hi unknown"}],
                },
            )
            self.assertEqual(st3, 404)
            self.assertIn(b"unknown-model", bd3)
            self.assertTrue(router.engine._backends[0].healthy)
            self.assertTrue(router.engine._backends[1].healthy)
        finally:
            primary_srv.shutdown()
            primary_srv.server_close()
            fallback_srv.shutdown()
            fallback_srv.server_close()


def run_self_tests(verbosity: int = 1) -> int:
    """Run all router tests cleanly via unittest."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFreeTPURouter)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_self_tests(verbosity=2))
