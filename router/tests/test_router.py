#!/usr/bin/env python3
"""Offline end-to-end integration and unit test suite for FreeTPU Router v0."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.config import BackendConfig, RouterConfig, load_config  # noqa: E402
from router.freetpu_router import (  # noqa: E402
    FreeTPURouterServer,
    MAX_RR_CURSORS,
    _apply_cli_overrides,
    _build_cli_parser,
    _join_backend_url,
    _open_http_connection,
    _safe_nonneg_int,
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
    evaluate_tpu_readiness,
    parse_prometheus_text,
    scrape_tpu_exporter,
    summarize_tpu_samples,
)


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
            if self.server.status_code >= 500:
                self._send_json(
                    self.server.status_code,
                    {"error": {"message": "worker unhealthy"}},
                )
                return
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
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

        # 1. Partial line exceeding max_sse_pre_commit_bytes (1024) with ZERO newlines -> fail over.
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
        self.assertEqual(status, 200)
        self.assertEqual(hdrs["x-freetpu-backend"], "good-nonl-backup")
        self.assertEqual(hdrs["x-freetpu-failovers"], "1")
        self.assertIn("good-nonl-backup", body.decode("utf-8"))

        # 2. Slow trickle without newlines (1 byte every 0.05s) hits 0.30s elapsed deadline -> fail over.
        router.engine._backends[0].cooldown_until_monotonic = 0.0
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

        # Explicit OpenAI usage chunk overrides earlier timings and is not overwritten later.
        worker.custom_sse_chunks = timings_chunks + [
            {
                "id": "chatcmpl-timings",
                "object": "chat.completion.chunk",
                "model": "m-timings",
                "choices": [],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15},
            }
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
        self.assertEqual(captured[1].completion_tokens, 15)

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

        huge_a = "huge-model-a-" + ("x" * 32768)
        huge_b = "huge-model-b-" + ("y" * 32768)
        seen: Dict[str, List[str]] = {huge_a: [], huge_b: []}
        for model_name in (huge_a, huge_b, huge_a, huge_b):
            status, hdrs, _ = _http_request(
                rport,
                "POST",
                "/v1/chat/completions",
                body={"model": model_name, "messages": [{"role": "user", "content": "rr"}]},
            )
            self.assertEqual(status, 200)
            seen[model_name].append(hdrs["x-freetpu-backend"])

        self.assertEqual(seen[huge_a], ["wc-hash-a", "wc-hash-b"])
        self.assertEqual(seen[huge_b], ["wc-hash-a", "wc-hash-b"])
        self.assertEqual(len(router.engine._rr_cursors), 2)
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
                    self.assertIn("upstream_stream_interrupted", partial)
                finally:
                    conn.close()

                self.assertEqual(router.engine._backends[0].inflight, 0)
                self.assertGreater(
                    router.engine._backends[0].cooldown_until_monotonic,
                    time.monotonic(),
                )
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

        # Case 1: Oversized non-streaming response (chunked and Content-Length) fails over to backup.
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
                self.assertEqual(status, 200)
                self.assertEqual(hdrs["x-freetpu-backend"], "healthy-backup")
                self.assertEqual(hdrs["x-freetpu-failovers"], "1")
                self.assertIn("reply-from:healthy-backup", body.decode("utf-8"))
                self.assertEqual(router.engine._backends[0].inflight, 0)
                self.assertGreater(
                    router.engine._backends[0].cooldown_until_monotonic,
                    time.monotonic(),
                )

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


def run_self_tests(verbosity: int = 1) -> int:
    """Run all router tests cleanly via unittest."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFreeTPURouter)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_self_tests(verbosity=2))
