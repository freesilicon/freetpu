#!/usr/bin/env python3
"""FreeTPU Router v0 — OpenAI-compatible router for heterogeneous TPU silicon.

Routes `/v1/chat/completions`, `/v1/completions`, and `/v1/models` across wired
Pixel Tensor G5 llama-server clusters, local CPU/GPU/Mac hosts, and Cloud TPU
v5e/v6e workers with zero third-party dependencies (Python standard library only).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import ssl
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.config import (  # noqa: E402
    BackendConfig,
    RouterConfig,
    VALID_POLICIES,
    load_config,
    sanitize_url,
)
from router.hooks import (  # noqa: E402
    AllowAllAuthHook,
    AuthContext,
    AuthHook,
    BackendIdleEvent,
    BearerTokenAuthHook,
    JsonlUsageMeterHook,
    NoOpScalerHook,
    NoOpUsageMeterHook,
    QueuePressureEvent,
    ScalerHook,
    UsageEvent,
    UsageMeterHook,
    load_hook_from_spec,
)
from router.metrics import RouterMetrics  # noqa: E402
from router.tpu_telemetry import (  # noqa: E402
    TpuExporterTelemetry,
    evaluate_tpu_readiness,
    scrape_tpu_exporter,
)

__version__ = "0.1.0"

RETRYABLE_STATUS_CODES = (502, 503, 504)


def _join_backend_url(base_url: str, endpoint: str) -> Tuple[str, str, int, str]:
    parsed = urlsplit(base_url)
    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    if endpoint.startswith("/v1/") and base_path.endswith("/v1"):
        full_path = base_path + endpoint[3:]
    elif endpoint == "/v1" and base_path.endswith("/v1"):
        full_path = base_path
    else:
        full_path = base_path + endpoint
    return scheme, host, port, full_path or "/"


def _open_http_connection(
    scheme: str,
    host: str,
    port: int,
    timeout_s: float,
) -> http.client.HTTPConnection:
    if scheme == "https":
        return http.client.HTTPSConnection(
            host,
            port,
            timeout=timeout_s,
            context=ssl.create_default_context(),
        )
    return http.client.HTTPConnection(host, port, timeout=timeout_s)


def _estimate_prompt_tokens(payload: Mapping[str, Any]) -> int:
    total_chars = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        total_chars += len(part["text"])
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        total_chars += len(prompt)
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, str):
                total_chars += len(item)
    if total_chars == 0:
        return 0
    return max(1, (total_chars + 3) // 4)


def _extract_sse_delta_text(obj: Mapping[str, Any]) -> str:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    delta = first.get("delta")
    if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
        return delta["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


@dataclass
class BackendState:
    """Mutable runtime health, latency, model inventory, and TPU telemetry for a backend."""

    config: BackendConfig
    healthy: bool = True
    unhealthy_reason: Optional[str] = None
    cooldown_until_monotonic: float = 0.0
    discovered_models: List[str] = field(default_factory=list)
    discovery_succeeded: bool = False
    ewma_latency_ms: Optional[float] = None
    last_probe_rtt_ms: Optional[float] = None
    last_probe_unix: Optional[float] = None
    last_used_monotonic: float = field(default_factory=time.monotonic)
    idle_notified: bool = False
    tpu_telemetry: Optional[TpuExporterTelemetry] = None
    inflight: int = 0
    total_requests: int = 0
    total_failures: int = 0

    def __post_init__(self) -> None:
        if self.ewma_latency_ms is None and self.config.latency_hint_ms is not None:
            self.ewma_latency_ms = self.config.latency_hint_ms

    def effective_models(self) -> List[str]:
        if self.config.discover_models and self.discovery_succeeded:
            if "*" in self.config.models and not self.discovered_models:
                return ["*"]
            return list(self.discovered_models)
        return list(self.config.models)

    def supports_model(self, model: str) -> bool:
        models = self.effective_models()
        return "*" in models or model in models

    def effective_latency_ms(self) -> float:
        if self.ewma_latency_ms is not None:
            return self.ewma_latency_ms
        if self.last_probe_rtt_ms is not None:
            return self.last_probe_rtt_ms
        return 50.0

    def hbm_ratio_or_zero(self) -> float:
        if (
            self.tpu_telemetry is not None
            and self.tpu_telemetry.max_hbm_utilization_ratio is not None
        ):
            return self.tpu_telemetry.max_hbm_utilization_ratio
        return 0.0

    def is_eligible(self, now_monotonic: float) -> bool:
        if not self.config.enabled:
            return False
        if not self.healthy:
            return False
        if now_monotonic < self.cooldown_until_monotonic:
            return False
        if (
            self.config.max_inflight is not None
            and self.inflight >= self.config.max_inflight
        ):
            return False
        return True

    def record_latency(self, latency_ms: float) -> None:
        if self.ewma_latency_ms is None:
            self.ewma_latency_ms = latency_ms
        else:
            self.ewma_latency_ms = (0.3 * latency_ms) + (0.7 * self.ewma_latency_ms)

    def snapshot(self, now_monotonic: Optional[float] = None) -> Dict[str, Any]:
        now_m = now_monotonic if now_monotonic is not None else time.monotonic()
        in_cooldown = now_m < self.cooldown_until_monotonic
        return {
            "id": self.config.id,
            "type": self.config.type,
            "local": self.config.local,
            "spot": self.config.spot,
            "enabled": self.config.enabled,
            "base_url": sanitize_url(self.config.base_url),
            "healthy": self.healthy and not in_cooldown,
            "unhealthy_reason": self.unhealthy_reason,
            "in_cooldown": in_cooldown,
            "inflight": self.inflight,
            "models": self.effective_models(),
            "latency_ms": round(self.effective_latency_ms(), 3)
            if (self.ewma_latency_ms is not None or self.last_probe_rtt_ms is not None)
            else None,
            "last_probe_rtt_ms": round(self.last_probe_rtt_ms, 3)
            if self.last_probe_rtt_ms is not None
            else None,
            "cost_per_1m_prompt_usd": self.config.cost_per_1m_prompt_usd,
            "cost_per_1m_completion_usd": self.config.cost_per_1m_completion_usd,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "tpu_telemetry": (
                self.tpu_telemetry.to_dict() if self.tpu_telemetry is not None else None
            ),
            "metadata": dict(self.config.metadata),
        }


def build_hooks_from_config(
    config: RouterConfig,
    *,
    auth_hook: Optional[AuthHook] = None,
    usage_hook: Optional[UsageMeterHook] = None,
    scaler_hook: Optional[ScalerHook] = None,
) -> Tuple[AuthHook, UsageMeterHook, ScalerHook]:
    """Instantiate configured hooks unless explicitly supplied by caller."""
    resolved_auth = auth_hook
    if resolved_auth is None:
        auth_cfg = config.auth
        spec = auth_cfg.get("hook")
        if spec:
            resolved_auth = load_hook_from_spec(str(spec), auth_cfg.get("kwargs"))
        elif auth_cfg.get("type") == "bearer":
            tokens: List[str] = [str(t) for t in auth_cfg.get("tokens") or [] if t]
            for env_name in auth_cfg.get("tokens_env") or []:
                import os

                val = os.environ.get(str(env_name), "").strip()
                if val:
                    tokens.append(val)
            exempt = auth_cfg.get("exempt_paths") or ("/health",)
            resolved_auth = BearerTokenAuthHook(tokens, exempt_paths=exempt)
        else:
            resolved_auth = AllowAllAuthHook()

    resolved_usage = usage_hook
    if resolved_usage is None:
        usage_cfg = config.usage_meter
        spec = usage_cfg.get("hook")
        if spec:
            resolved_usage = load_hook_from_spec(str(spec), usage_cfg.get("kwargs"))
        elif usage_cfg.get("type") == "jsonl" or usage_cfg.get("jsonl_path"):
            jsonl_path = usage_cfg.get("jsonl_path") or "freetpu_router_usage.jsonl"
            resolved_usage = JsonlUsageMeterHook(str(jsonl_path))
        else:
            resolved_usage = NoOpUsageMeterHook()

    resolved_scaler = scaler_hook
    if resolved_scaler is None:
        scaler_cfg = config.scaler
        spec = scaler_cfg.get("hook")
        if spec:
            resolved_scaler = load_hook_from_spec(str(spec), scaler_cfg.get("kwargs"))
        else:
            resolved_scaler = NoOpScalerHook()

    return resolved_auth, resolved_usage, resolved_scaler


class RouterEngine:
    """Core routing, health polling, TPU telemetry, and failover engine."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        auth_hook: Optional[AuthHook] = None,
        usage_hook: Optional[UsageMeterHook] = None,
        scaler_hook: Optional[ScalerHook] = None,
    ) -> None:
        self.config = config
        self.auth_hook, self.usage_hook, self.scaler_hook = build_hooks_from_config(
            config,
            auth_hook=auth_hook,
            usage_hook=usage_hook,
            scaler_hook=scaler_hook,
        )
        self.metrics = RouterMetrics()
        self._lock = threading.RLock()
        self._backends: List[BackendState] = [
            BackendState(config=b) for b in config.backends
        ]
        self._rr_cursors: Dict[str, int] = {}
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def start_background_poller(self) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="freetpu-router-poller",
            daemon=True,
        )
        self._poll_thread.start()

    def stop_background_poller(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.config.health_poll_interval_s):
            try:
                self.refresh_all_backends()
                self.check_idle_backends()
            except Exception:
                # Keep poller resilient across transient probe errors.
                pass

    def refresh_all_backends(self) -> None:
        for state in self._backends:
            if state.config.enabled:
                self.probe_backend(state)

    def check_idle_backends(self) -> None:
        now_m = time.monotonic()
        threshold = self.config.backend_idle_timeout_s
        to_notify: List[BackendIdleEvent] = []
        with self._lock:
            for state in self._backends:
                if not state.config.enabled or state.inflight > 0:
                    continue
                idle_s = now_m - state.last_used_monotonic
                if idle_s >= threshold and not state.idle_notified:
                    state.idle_notified = True
                    to_notify.append(
                        BackendIdleEvent(
                            backend_id=state.config.id,
                            backend_type=state.config.type,
                            spot=state.config.spot,
                            idle_seconds=idle_s,
                        )
                    )
        for event in to_notify:
            try:
                self.scaler_hook.on_backend_idle(event)
            except Exception:
                pass

    def _http_get_json(
        self,
        backend: BackendConfig,
        endpoint: str,
        timeout_s: float,
    ) -> Tuple[int, Optional[Dict[str, Any]], float, Optional[str]]:
        scheme, host, port, path = _join_backend_url(backend.base_url, endpoint)
        headers: Dict[str, str] = {"Accept": "application/json"}
        api_key = backend.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.monotonic()
        conn: Optional[http.client.HTTPConnection] = None
        try:
            conn = _open_http_connection(scheme, host, port, timeout_s)
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            rtt_ms = (time.monotonic() - t0) * 1000.0
            parsed_json: Optional[Dict[str, Any]] = None
            if raw:
                try:
                    loaded = json.loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(loaded, dict):
                        parsed_json = loaded
                except json.JSONDecodeError:
                    parsed_json = None
            return resp.status, parsed_json, rtt_ms, None
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            rtt_ms = (time.monotonic() - t0) * 1000.0
            return 0, None, rtt_ms, str(exc)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass

    def probe_backend(self, state: BackendState) -> None:
        """Probe backend HTTP readiness, `/v1/models`, and optional `tpu-exporter`."""
        timeout_s = self.config.probe_timeout_s
        probe_ok = False
        probe_err: Optional[str] = None
        rtt_ms: Optional[float] = None
        discovered: Optional[List[str]] = None

        if state.config.health_path:
            status, _, rtt_ms, err = self._http_get_json(
                state.config, state.config.health_path, timeout_s
            )
            if 200 <= status < 300:
                probe_ok = True
            else:
                probe_err = err or f"health_http_{status}"
            if probe_ok and state.config.discover_models:
                m_status, m_json, _, _ = self._http_get_json(
                    state.config, "/v1/models", timeout_s
                )
                if 200 <= m_status < 300 and m_json and isinstance(m_json.get("data"), list):
                    discovered = [
                        str(item["id"])
                        for item in m_json["data"]
                        if isinstance(item, Mapping) and item.get("id")
                    ]
        else:
            status, m_json, rtt_ms, err = self._http_get_json(
                state.config, "/v1/models", timeout_s
            )
            if 200 <= status < 300:
                probe_ok = True
                if m_json and isinstance(m_json.get("data"), list):
                    discovered = [
                        str(item["id"])
                        for item in m_json["data"]
                        if isinstance(item, Mapping) and item.get("id")
                    ]
            else:
                probe_err = err or f"probe_http_{status}"

        tpu_tele: Optional[TpuExporterTelemetry] = None
        tpu_ready = True
        tpu_reason: Optional[str] = None
        if state.config.tpu_exporter_url:
            tpu_tele = scrape_tpu_exporter(
                state.config.tpu_exporter_url, timeout_s=timeout_s
            )
            tpu_ready, tpu_reason = evaluate_tpu_readiness(
                tpu_tele,
                max_hbm_utilization=state.config.max_hbm_utilization,
                max_duty_cycle_percent=state.config.max_duty_cycle_percent,
            )

        now_m = time.monotonic()
        with self._lock:
            state.last_probe_unix = time.time()
            if probe_ok and rtt_ms is not None:
                state.last_probe_rtt_ms = rtt_ms
            if discovered is not None:
                state.discovered_models = discovered
                state.discovery_succeeded = True
            if tpu_tele is not None:
                state.tpu_telemetry = tpu_tele

            if not probe_ok:
                state.healthy = False
                state.unhealthy_reason = probe_err
            elif not tpu_ready:
                state.healthy = False
                state.unhealthy_reason = tpu_reason
            else:
                state.healthy = True
                state.unhealthy_reason = None
                if now_m >= state.cooldown_until_monotonic:
                    state.cooldown_until_monotonic = 0.0

    def mark_backend_failed(self, state: BackendState, reason: str) -> None:
        now_m = time.monotonic()
        with self._lock:
            state.total_failures += 1
            state.unhealthy_reason = reason
            if self.config.failure_cooldown_s > 0:
                state.cooldown_until_monotonic = now_m + self.config.failure_cooldown_s
            else:
                state.healthy = False

    def select_candidates(
        self,
        model: str,
        *,
        policy: Optional[str] = None,
    ) -> List[BackendState]:
        active_policy = policy or self.config.policy
        if active_policy not in VALID_POLICIES:
            active_policy = self.config.policy

        now_m = time.monotonic()
        with self._lock:
            eligible = [
                b
                for b in self._backends
                if b.is_eligible(now_m) and b.supports_model(model)
            ]
            if not eligible:
                return []

            if active_policy == "cheapest":
                eligible.sort(
                    key=lambda b: (
                        b.config.blended_cost_per_1m_tokens(),
                        0 if b.config.local else 1,
                        b.effective_latency_ms(),
                        b.hbm_ratio_or_zero(),
                        b.inflight,
                        b.config.priority,
                        b.config.id,
                    )
                )
                return eligible

            if active_policy == "local_first":
                eligible.sort(
                    key=lambda b: (
                        0 if b.config.local else 1,
                        b.config.priority,
                        b.config.blended_cost_per_1m_tokens(),
                        b.effective_latency_ms(),
                        b.hbm_ratio_or_zero(),
                        b.inflight,
                        b.config.id,
                    )
                )
                return eligible

            if active_policy == "lowest_latency":
                eligible.sort(
                    key=lambda b: (
                        b.effective_latency_ms(),
                        b.inflight,
                        b.hbm_ratio_or_zero(),
                        0 if b.config.local else 1,
                        b.config.blended_cost_per_1m_tokens(),
                        b.config.id,
                    )
                )
                return eligible

            # round_robin
            eligible.sort(key=lambda b: (b.config.priority, b.config.id))
            cursor = self._rr_cursors.get(model, 0)
            self._rr_cursors[model] = (cursor + 1) % len(eligible)
            offset = cursor % len(eligible)
            return eligible[offset:] + eligible[:offset]

    def aggregate_models(self) -> Dict[str, Any]:
        """Aggregate unique models served across currently healthy backends."""
        now_m = time.monotonic()
        created = int(time.time())
        models_map: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for b in self._backends:
                if not b.is_eligible(now_m):
                    continue
                for model_id in b.effective_models():
                    if model_id == "*":
                        continue
                    entry = models_map.get(model_id)
                    if entry is None:
                        models_map[model_id] = {
                            "id": model_id,
                            "object": "model",
                            "created": created,
                            "owned_by": b.config.type,
                            "freetpu_backends": [b.config.id],
                        }
                    else:
                        entry["freetpu_backends"].append(b.config.id)
        data = [models_map[k] for k in sorted(models_map.keys())]
        return {"object": "list", "data": data}

    def health_payload(self) -> Tuple[int, Dict[str, Any]]:
        now_m = time.monotonic()
        with self._lock:
            snaps = [b.snapshot(now_m) for b in self._backends]
        healthy_count = sum(1 for s in snaps if s["healthy"])
        status_str = "ok" if healthy_count > 0 else "degraded"
        http_code = 200 if healthy_count > 0 else 503
        payload = {
            "status": status_str,
            "version": __version__,
            "policy": self.config.policy,
            "healthy_backends": healthy_count,
            "total_backends": len(snaps),
            "active_inflight": self.metrics.total_inflight(),
            "backends": snaps,
        }
        return http_code, payload

    def metrics_text(self) -> str:
        now_m = time.monotonic()
        with self._lock:
            snaps = [b.snapshot(now_m) for b in self._backends]
        return self.metrics.render_prometheus(snaps)

    def _emit_queue_pressure(
        self,
        *,
        model: str,
        active_inflight: int,
        healthy_backends: int,
        reason: str,
    ) -> None:
        event = QueuePressureEvent(
            model=model,
            active_inflight=active_inflight,
            healthy_backends=healthy_backends,
            total_backends=len(self._backends),
            reason=reason,
        )
        try:
            self.scaler_hook.on_queue_pressure(event)
        except Exception:
            pass

    def _record_usage(
        self,
        *,
        request_id: str,
        model: str,
        backend: BackendState,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        ttft_ms: Optional[float],
        stream: bool,
        failovers: int,
        principal: str,
        status_code: int,
    ) -> None:
        est_cost = backend.config.estimate_cost_usd(prompt_tokens, completion_tokens)
        self.metrics.record_request(
            model=model,
            backend_id=backend.config.id,
            status=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            estimated_cost_usd=est_cost,
        )
        event = UsageEvent(
            request_id=request_id,
            model=model,
            backend_id=backend.config.id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 3),
            estimated_cost_usd=round(est_cost, 10),
            ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
            stream=stream,
            failovers=failovers,
            principal=principal,
            status_code=status_code,
        )
        try:
            self.usage_hook.record_usage(event)
        except Exception:
            pass


def _openai_error_body(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> bytes:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


class FreeTPURouterServer(ThreadingHTTPServer):
    """Multi-threaded stdlib HTTP server hosting `RouterEngine`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        engine: RouterEngine,
        *,
        verbose: bool = False,
    ) -> None:
        self.engine = engine
        self.verbose = verbose
        super().__init__(server_address, _RouterRequestHandler)


class _RouterRequestHandler(BaseHTTPRequestHandler):
    server: FreeTPURouterServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.wfile.flush()

    def _send_json(
        self,
        status: int,
        payload: Any,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self._send_bytes(status, "application/json", body, extra_headers=extra_headers)

    def _authorize(self, path: str, method: str) -> Optional[AuthContext]:
        hdrs = {k: v for k, v in self.headers.items()}
        ctx = self.server.engine.auth_hook.authenticate(hdrs, path, method)
        if not ctx.authorized:
            self._send_bytes(
                ctx.status_code,
                "application/json",
                _openai_error_body(
                    ctx.error_message or "Unauthorized",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
            )
            return None
        return ctx

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, X-FreeTPU-Policy",
        )
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        ctx = self._authorize(path, "GET")
        if ctx is None:
            return

        if path == "/health":
            code, payload = self.server.engine.health_payload()
            self._send_json(code, payload)
            return

        if path == "/metrics":
            body = self.server.engine.metrics_text().encode("utf-8")
            self._send_bytes(200, "text/plain; version=0.0.4; charset=utf-8", body)
            return

        if path == "/v1/models":
            self._send_json(200, self.server.engine.aggregate_models())
            return

        self._send_bytes(
            404,
            "application/json",
            _openai_error_body(f"Unknown route GET {path}", code="not_found"),
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        ctx = self._authorize(path, "POST")
        if ctx is None:
            return

        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_bytes(
                404,
                "application/json",
                _openai_error_body(f"Unknown route POST {path}", code="not_found"),
            )
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_bytes(
                400,
                "application/json",
                _openai_error_body("Invalid Content-Length header"),
            )
            return

        if content_len <= 0:
            self._send_bytes(
                400,
                "application/json",
                _openai_error_body("Request body must be non-empty JSON"),
            )
            return

        raw_req = self.rfile.read(content_len)
        try:
            req_payload = json.loads(raw_req.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_bytes(
                400,
                "application/json",
                _openai_error_body(f"Malformed JSON request: {exc}"),
            )
            return

        if not isinstance(req_payload, dict):
            self._send_bytes(
                400,
                "application/json",
                _openai_error_body("Request JSON body must be an object"),
            )
            return

        model = str(req_payload.get("model") or "").strip()
        if not model:
            self._send_bytes(
                400,
                "application/json",
                _openai_error_body("Field 'model' is required"),
            )
            return

        self._handle_inference(path, raw_req, req_payload, model, ctx)

    def _handle_inference(
        self,
        endpoint: str,
        raw_req: bytes,
        req_payload: Dict[str, Any],
        model: str,
        auth_ctx: AuthContext,
    ) -> None:
        engine = self.server.engine
        stream = bool(req_payload.get("stream", False))
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        req_start = time.monotonic()
        policy_override = self.headers.get("X-FreeTPU-Policy")
        if policy_override:
            policy_override = policy_override.strip()

        candidates = engine.select_candidates(model, policy=policy_override)
        if not candidates:
            engine._emit_queue_pressure(
                model=model,
                active_inflight=engine.metrics.total_inflight(),
                healthy_backends=0,
                reason="no_healthy_backend",
            )
            self._send_bytes(
                503,
                "application/json",
                _openai_error_body(
                    f"No healthy backend available for model {model!r}",
                    error_type="service_unavailable",
                    code="no_healthy_backend",
                ),
                extra_headers={"X-FreeTPU-Request-Id": req_id},
            )
            return

        total_inflight_now = engine.metrics.total_inflight() + 1
        if total_inflight_now >= engine.config.queue_pressure_threshold:
            engine._emit_queue_pressure(
                model=model,
                active_inflight=total_inflight_now,
                healthy_backends=len(candidates),
                reason="inflight_threshold_exceeded",
            )

        est_prompt_tokens = _estimate_prompt_tokens(req_payload)
        failovers = 0
        last_error_status = 503
        last_error_msg = "All candidate backends failed"

        for idx, backend in enumerate(candidates):
            attempt_start = time.monotonic()
            engine.metrics.inc_inflight(backend.config.id)
            with engine._lock:
                backend.inflight += 1
                backend.total_requests += 1
                backend.last_used_monotonic = attempt_start
                backend.idle_notified = False

            conn: Optional[http.client.HTTPConnection] = None
            try:
                scheme, host, port, upstream_path = _join_backend_url(
                    backend.config.base_url, endpoint
                )
                timeout_s = backend.config.timeout_s or engine.config.request_timeout_s
                conn = _open_http_connection(scheme, host, port, timeout_s)
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream" if stream else "application/json",
                    "X-FreeTPU-Request-Id": req_id,
                }
                api_key = backend.config.resolved_api_key()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                conn.request("POST", upstream_path, body=raw_req, headers=headers)
                resp = conn.getresponse()

                if resp.status in RETRYABLE_STATUS_CODES:
                    reason = f"http_{resp.status}"
                    last_error_status = resp.status
                    last_error_msg = f"Backend {backend.config.id!r} returned {resp.status}"
                    engine.mark_backend_failed(backend, reason)
                    if idx + 1 < len(candidates):
                        failovers += 1
                        engine.metrics.record_failover(
                            model=model,
                            from_backend=backend.config.id,
                            to_backend=candidates[idx + 1].config.id,
                            reason=reason,
                        )
                        continue
                    err_body = resp.read()
                    self._send_bytes(
                        resp.status,
                        resp.getheader("Content-Type") or "application/json",
                        err_body
                        or _openai_error_body(
                            last_error_msg,
                            error_type="service_unavailable",
                            code="backend_unavailable",
                        ),
                        extra_headers={
                            "X-FreeTPU-Backend": backend.config.id,
                            "X-FreeTPU-Failovers": str(failovers),
                            "X-FreeTPU-Request-Id": req_id,
                        },
                    )
                    return

                if stream and 200 <= resp.status < 300:
                    completed = self._proxy_sse_stream(
                        resp=resp,
                        backend=backend,
                        model=model,
                        req_id=req_id,
                        req_start=req_start,
                        attempt_start=attempt_start,
                        failovers=failovers,
                        est_prompt_tokens=est_prompt_tokens,
                        principal=auth_ctx.principal,
                        has_next_candidate=(idx + 1 < len(candidates)),
                    )
                    if completed:
                        return
                    # Pre-first-token stream failure: fail over to next candidate.
                    last_error_status = 502
                    last_error_msg = (
                        f"Backend {backend.config.id!r} aborted stream before first token"
                    )
                    engine.mark_backend_failed(backend, "stream_aborted_before_first_token")
                    if idx + 1 < len(candidates):
                        failovers += 1
                        engine.metrics.record_failover(
                            model=model,
                            from_backend=backend.config.id,
                            to_backend=candidates[idx + 1].config.id,
                            reason="stream_aborted_before_first_token",
                        )
                        continue
                    self._send_bytes(
                        502,
                        "application/json",
                        _openai_error_body(
                            last_error_msg,
                            error_type="bad_gateway",
                            code="upstream_stream_error",
                        ),
                        extra_headers={
                            "X-FreeTPU-Backend": backend.config.id,
                            "X-FreeTPU-Failovers": str(failovers),
                            "X-FreeTPU-Request-Id": req_id,
                        },
                    )
                    return

                # Non-streaming response (or non-retryable error status such as 4xx)
                raw_resp = resp.read()
                latency_ms = (time.monotonic() - req_start) * 1000.0
                attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                content_type = resp.getheader("Content-Type") or "application/json"

                prompt_tokens = est_prompt_tokens
                completion_tokens = 0
                if 200 <= resp.status < 300:
                    with engine._lock:
                        backend.record_latency(attempt_ms)
                    try:
                        parsed_resp = json.loads(raw_resp.decode("utf-8", errors="replace"))
                        if isinstance(parsed_resp, Mapping):
                            usage = parsed_resp.get("usage")
                            if isinstance(usage, Mapping):
                                if usage.get("prompt_tokens") is not None:
                                    prompt_tokens = int(usage["prompt_tokens"])
                                if usage.get("completion_tokens") is not None:
                                    completion_tokens = int(usage["completion_tokens"])
                    except json.JSONDecodeError:
                        pass
                else:
                    prompt_tokens = 0

                engine._record_usage(
                    request_id=req_id,
                    model=model,
                    backend=backend,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    ttft_ms=latency_ms if 200 <= resp.status < 300 else None,
                    stream=False,
                    failovers=failovers,
                    principal=auth_ctx.principal,
                    status_code=resp.status,
                )
                self._send_bytes(
                    resp.status,
                    content_type,
                    raw_resp,
                    extra_headers={
                        "X-FreeTPU-Backend": backend.config.id,
                        "X-FreeTPU-Failovers": str(failovers),
                        "X-FreeTPU-Request-Id": req_id,
                    },
                )
                return

            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                reason = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
                last_error_status = 504 if isinstance(exc, TimeoutError) else 502
                last_error_msg = f"Backend {backend.config.id!r} failed: {exc}"
                engine.mark_backend_failed(backend, reason)
                if idx + 1 < len(candidates):
                    failovers += 1
                    engine.metrics.record_failover(
                        model=model,
                        from_backend=backend.config.id,
                        to_backend=candidates[idx + 1].config.id,
                        reason=reason,
                    )
                    continue
                self._send_bytes(
                    last_error_status,
                    "application/json",
                    _openai_error_body(
                        last_error_msg,
                        error_type="service_unavailable",
                        code=reason,
                    ),
                    extra_headers={
                        "X-FreeTPU-Backend": backend.config.id,
                        "X-FreeTPU-Failovers": str(failovers),
                        "X-FreeTPU-Request-Id": req_id,
                    },
                )
                return
            finally:
                with engine._lock:
                    backend.inflight = max(0, backend.inflight - 1)
                engine.metrics.dec_inflight(backend.config.id)
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass

        self._send_bytes(
            last_error_status,
            "application/json",
            _openai_error_body(
                last_error_msg,
                error_type="service_unavailable",
                code="all_backends_failed",
            ),
            extra_headers={"X-FreeTPU-Request-Id": req_id},
        )

    def _proxy_sse_stream(
        self,
        *,
        resp: http.client.HTTPResponse,
        backend: BackendState,
        model: str,
        req_id: str,
        req_start: float,
        attempt_start: float,
        failovers: int,
        est_prompt_tokens: int,
        principal: str,
        has_next_candidate: bool,
    ) -> bool:
        """Buffer until first SSE chunk before committing 200 OK; then stream live.

        Returns True if headers/stream were committed to the downstream client,
        or False if upstream failed before delivering the first token/chunk.
        """
        engine = self.server.engine
        pre_commit_lines: List[bytes] = []
        first_chunk_time: Optional[float] = None

        try:
            while True:
                line = resp.fp.readline()
                if not line:
                    if not pre_commit_lines:
                        return False
                    break
                pre_commit_lines.append(line)
                stripped = line.strip()
                if stripped.startswith(b"data:") or stripped:
                    first_chunk_time = time.monotonic()
                    break
        except (OSError, TimeoutError, http.client.HTTPException):
            if has_next_candidate:
                return False
            raise

        # Upstream delivered first SSE line cleanly: commit downstream response headers.
        ttft_ms: Optional[float] = (
            (first_chunk_time - req_start) * 1000.0 if first_chunk_time else None
        )
        first_content_ttft_ms: Optional[float] = None
        prompt_tokens = est_prompt_tokens
        completion_tokens = 0
        saw_explicit_usage = False

        def inspect_sse_line(raw_line: bytes) -> None:
            nonlocal first_content_ttft_ms, prompt_tokens, completion_tokens, saw_explicit_usage
            stripped = raw_line.strip()
            if not stripped.startswith(b"data:"):
                return
            payload_bytes = stripped[5:].strip()
            if not payload_bytes or payload_bytes == b"[DONE]":
                return
            try:
                obj = json.loads(payload_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                return
            if not isinstance(obj, Mapping):
                return
            delta_txt = _extract_sse_delta_text(obj)
            if delta_txt:
                if first_content_ttft_ms is None:
                    first_content_ttft_ms = (time.monotonic() - req_start) * 1000.0
                if not saw_explicit_usage:
                    completion_tokens += 1
            usage = obj.get("usage")
            if isinstance(usage, Mapping):
                saw_explicit_usage = True
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])

        self.send_response(resp.status)
        self.send_header(
            "Content-Type",
            resp.getheader("Content-Type") or "text/event-stream; charset=utf-8",
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.send_header("X-FreeTPU-Backend", backend.config.id)
        self.send_header("X-FreeTPU-Failovers", str(failovers))
        self.send_header("X-FreeTPU-Request-Id", req_id)
        self.end_headers()

        try:
            for buffered in pre_commit_lines:
                inspect_sse_line(buffered)
                self.wfile.write(buffered)
            self.wfile.flush()

            while True:
                line = resp.fp.readline()
                if not line:
                    break
                inspect_sse_line(line)
                self.wfile.write(line)
                if line in (b"\n", b"\r\n"):
                    self.wfile.flush()
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client disconnected or upstream closed mid-stream after first token.
            pass
        finally:
            total_ms = (time.monotonic() - req_start) * 1000.0
            attempt_ms = (time.monotonic() - attempt_start) * 1000.0
            effective_ttft = first_content_ttft_ms if first_content_ttft_ms is not None else ttft_ms
            with engine._lock:
                backend.record_latency(attempt_ms)
            engine._record_usage(
                request_id=req_id,
                model=model,
                backend=backend,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=total_ms,
                ttft_ms=effective_ttft,
                stream=True,
                failovers=failovers,
                principal=principal,
                status_code=resp.status,
            )

        return True


def create_router_server(
    config: RouterConfig,
    *,
    auth_hook: Optional[AuthHook] = None,
    usage_hook: Optional[UsageMeterHook] = None,
    scaler_hook: Optional[ScalerHook] = None,
    probe_on_start: bool = True,
    verbose: bool = False,
) -> FreeTPURouterServer:
    """Create a configured `FreeTPURouterServer` ready to call `.serve_forever()`."""
    engine = RouterEngine(
        config,
        auth_hook=auth_hook,
        usage_hook=usage_hook,
        scaler_hook=scaler_hook,
    )
    if probe_on_start:
        engine.refresh_all_backends()
    return FreeTPURouterServer((config.host, config.port), engine, verbose=verbose)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FreeTPU Router v0 — OpenAI-compatible router for heterogeneous "
            "Pixel Tensor G5, Mac/CPU/GPU, and Cloud TPU backends."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to router JSON configuration file (see router/examples/).",
    )
    parser.add_argument("--host", help="Override bind host (default from config or 127.0.0.1).")
    parser.add_argument("--port", type=int, help="Override bind port (default from config or 8080).")
    parser.add_argument(
        "--policy",
        choices=VALID_POLICIES,
        help="Override routing policy (cheapest, local_first, lowest_latency, round_robin).",
    )
    parser.add_argument(
        "--bearer-token",
        help="Optional bearer token required on API requests (configures BearerTokenAuthHook).",
    )
    parser.add_argument(
        "--usage-jsonl",
        type=Path,
        help="Optional JSONL path to record per-request UsageMeterHook events.",
    )
    parser.add_argument(
        "--no-initial-probe",
        action="store_true",
        help="Skip synchronous backend probe before binding the HTTP listener.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate --config JSON and exit 0 without starting the server.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the offline end-to-end integration test suite and exit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log HTTP access lines to stderr.")
    parser.add_argument("--version", action="version", version=f"freetpu-router {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        from router.tests.test_router import run_self_tests

        return run_self_tests(verbosity=2 if args.verbose else 1)

    if not args.config:
        parser.error("--config is required unless --self-test is passed")

    cfg = load_config(args.config)
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.policy is not None:
        cfg.policy = args.policy
    if args.bearer_token:
        cfg.auth = {"type": "bearer", "tokens": [args.bearer_token]}
    if args.usage_jsonl:
        cfg.usage_meter = {"type": "jsonl", "jsonl_path": str(args.usage_jsonl)}

    if args.check_config:
        print(
            f"Config OK: policy={cfg.policy}, backends={len(cfg.backends)}, "
            f"listen={cfg.host}:{cfg.port}"
        )
        return 0

    server = create_router_server(
        cfg,
        probe_on_start=not args.no_initial_probe,
        verbose=args.verbose,
    )
    server.engine.start_background_poller()
    bound_host, bound_port = server.server_address[:2]
    print(
        f"FreeTPU Router v{__version__} listening on http://{bound_host}:{bound_port}/v1 "
        f"(policy={cfg.policy}, backends={len(cfg.backends)})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.engine.stop_background_poller()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
