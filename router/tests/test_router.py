#!/usr/bin/env python3
"""Offline end-to-end integration and unit test suite for FreeTPU Router v0."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.config import BackendConfig, RouterConfig, load_config  # noqa: E402
from router.freetpu_router import (  # noqa: E402
    FreeTPURouterServer,
    create_router_server,
)
from router.hooks import (  # noqa: E402
    BackendIdleEvent,
    BearerTokenAuthHook,
    JsonlUsageMeterHook,
    QueuePressureEvent,
    ScalerHook,
    UsageEvent,
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
        prompt_tokens: int = 12,
        completion_tokens: int = 8,
    ) -> None:
        self.backend_name = backend_name
        self.models = models
        self.status_code = status_code
        self.latency_s = latency_s
        self.abort_stream_before_first_token = abort_stream_before_first_token
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests_seen: List[Dict[str, Any]] = []
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
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.server.abort_stream_before_first_token:
                self.wfile.flush()
                return
            chunks = [
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


def run_self_tests(verbosity: int = 1) -> int:
    """Run all router tests cleanly via unittest."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFreeTPURouter)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_self_tests(verbosity=2))
