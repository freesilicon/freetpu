#!/usr/bin/env python3
"""Offline end-to-end integration and unit test suite for FreeTPU Router v0."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
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
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests_seen: List[Dict[str, Any]] = []
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
        if self.path == "/v1/models":
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
    ) -> None:
        self.exporter_up = exporter_up
        self.health_status = health_status
        self.hbm_used_bytes = hbm_used_bytes
        self.hbm_total_bytes = hbm_total_bytes
        self.duty_cycle_percent = duty_cycle_percent
        self.generation = generation
        super().__init__(("127.0.0.1", 0), _MockTpuExporterHandler)

    @property
    def metrics_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/metrics"

    def render(self) -> str:
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
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
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
            'freetpu_router_requests_total{model="unknown-model",backend_id="none",status="503"} 1',
            metrics_txt,
        )
        self.assertIn(
            'freetpu_router_requests_total{model="m-503",backend_id="worker-503",status="503"} 1',
            metrics_txt,
        )
        self.assertIn(
            'freetpu_router_requests_total{model="m-dead",backend_id="worker-dead",status="502"} 1',
            metrics_txt,
        )


def run_self_tests(verbosity: int = 1) -> int:
    """Run all router tests cleanly via unittest."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFreeTPURouter)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_self_tests(verbosity=2))
