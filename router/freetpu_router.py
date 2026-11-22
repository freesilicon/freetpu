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
    BEARER_AUTH_ONLY_KEYS,
    BackendConfig,
    RouterConfig,
    VALID_AUTH_TYPES,
    VALID_POLICIES,
    VALID_URL_SCHEMES,
    is_loopback_host,
    load_config,
    sanitize_url,
    validate_http_url,
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
UNKNOWN_MODEL_LABEL = "_unknown"
OTHER_MODEL_LABEL = "_other"
MAX_METRIC_MODEL_LABELS = 256


def _join_backend_url(base_url: str, endpoint: str) -> Tuple[str, str, int, str]:
    validated = validate_http_url(base_url, field_name="backend base_url")
    parsed = urlsplit(validated)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
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
    normalized = scheme.lower()
    if normalized == "https":
        return http.client.HTTPSConnection(
            host,
            port,
            timeout=timeout_s,
            context=ssl.create_default_context(),
        )
    if normalized == "http":
        return http.client.HTTPConnection(host, port, timeout=timeout_s)
    raise ValueError(
        f"Unsupported backend URL scheme {scheme!r}; expected one of {VALID_URL_SCHEMES}"
    )


def _ensure_safe_credential_transport(
    backend: BackendConfig,
    scheme: str,
    host: str,
    api_key: Optional[str],
) -> None:
    """Refuse to transmit backend bearer tokens over plaintext HTTP to non-loopback hosts."""
    if (
        api_key
        and scheme.lower() == "http"
        and not is_loopback_host(host)
        and not backend.allow_insecure_http
    ):
        raise ValueError(
            f"Refusing to send bearer credentials for backend {backend.id!r} over "
            f"plaintext HTTP to non-loopback host {host!r}; use 'https://' or set "
            "'allow_insecure_http': true"
        )


def _expected_stream_choices(endpoint: str, req_payload: Mapping[str, Any]) -> int:
    raw_n = req_payload.get("n", 1)
    n_val = (
        raw_n
        if isinstance(raw_n, int) and not isinstance(raw_n, bool) and raw_n >= 1
        else 1
    )
    if endpoint == "/v1/completions":
        prompt = req_payload.get("prompt")
        if isinstance(prompt, list) and prompt:
            if all(isinstance(item, (str, list)) for item in prompt):
                return len(prompt) * n_val
    return n_val


class _BoundedSSEReader:
    """Incremental SSE line reader decoding HTTP framing via `HTTPResponse.read1()`.

    Enforces optional wall-clock deadlines and byte caps on every incremental
    read so partial lines without newlines cannot exceed the byte cap or stall
    past the deadline.
    """

    def __init__(
        self,
        resp: http.client.HTTPResponse,
        upstream_sock: Any,
        *,
        chunk_size: int = 8192,
    ) -> None:
        self._resp = resp
        self._sock = upstream_sock
        self._chunk_size = chunk_size
        self._buf = bytearray()
        self._eof = False

    def readline(
        self,
        *,
        deadline: Optional[float] = None,
        max_bytes: Optional[int] = None,
    ) -> Tuple[bytes, bool]:
        """Read one SSE line (`...\\n`) or EOF (`b""`).

        Returns `(line, exceeded_max_bytes)`. Raises `TimeoutError` if `deadline`
        expires before a newline or EOF arrives.
        """
        while True:
            nl_idx = self._buf.find(b"\n")
            if nl_idx != -1:
                end = nl_idx + 1
                if max_bytes is not None and end > max_bytes:
                    return b"", True
                out = bytes(self._buf[:end])
                del self._buf[:end]
                return out, False

            if max_bytes is not None and len(self._buf) > max_bytes:
                return b"", True

            if self._eof:
                out = bytes(self._buf)
                self._buf.clear()
                return out, False

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for first SSE output")
                if self._sock is not None:
                    try:
                        self._sock.settimeout(remaining)
                    except OSError:
                        pass

            to_read = self._chunk_size
            if max_bytes is not None:
                budget_left = (max_bytes - len(self._buf)) + 1
                if budget_left <= 0:
                    return b"", True
                to_read = min(to_read, budget_left)

            chunk = self._resp.read1(to_read)
            if not chunk:
                self._eof = True
                if self._buf:
                    if max_bytes is not None and len(self._buf) > max_bytes:
                        return b"", True
                    out = bytes(self._buf)
                    self._buf.clear()
                    return out, False
                return b"", False

            self._buf.extend(chunk)


def _estimate_prompt_tokens(payload: Mapping[str, Any]) -> int:
    total_chars = 0
    explicit_tokens = 0
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
            elif isinstance(item, int) and not isinstance(item, bool):
                explicit_tokens += 1
            elif isinstance(item, list):
                explicit_tokens += sum(
                    1
                    for tok in item
                    if isinstance(tok, int) and not isinstance(tok, bool)
                )
    char_tokens = max(1, (total_chars + 3) // 4) if total_chars > 0 else 0
    return explicit_tokens + char_tokens


def _extract_choice_output_text(obj: Mapping[str, Any]) -> str:
    """Extract all generated text and tool-call output across every choice."""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    parts: List[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        text_val = choice.get("text")
        if isinstance(text_val, str) and text_val:
            parts.append(text_val)
        for container_key in ("delta", "message"):
            container = choice.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for text_key in ("content", "refusal", "reasoning_content", "reasoning"):
                val = container.get(text_key)
                if isinstance(val, str) and val:
                    parts.append(val)
            func_call = container.get("function_call")
            if isinstance(func_call, Mapping):
                for fk in ("name", "arguments"):
                    fval = func_call.get(fk)
                    if isinstance(fval, str) and fval:
                        parts.append(fval)
            tool_calls = container.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, Mapping):
                        continue
                    fn = tc.get("function")
                    if isinstance(fn, Mapping):
                        for fk in ("name", "arguments"):
                            fval = fn.get(fk)
                            if isinstance(fval, str) and fval:
                                parts.append(fval)
    return "".join(parts)


def _extract_non_stream_usage(
    raw_resp: bytes,
    est_prompt_tokens: int,
) -> Tuple[int, int]:
    """Extract `(prompt_tokens, completion_tokens)` preferring upstream counts."""
    prompt_tokens = est_prompt_tokens
    completion_tokens = 0
    saw_prompt = False
    saw_completion = False
    try:
        parsed_resp = json.loads(raw_resp.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return prompt_tokens, completion_tokens
    if not isinstance(parsed_resp, Mapping):
        return prompt_tokens, completion_tokens

    usage = parsed_resp.get("usage")
    if isinstance(usage, Mapping):
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = int(usage["prompt_tokens"])
            saw_prompt = True
        if usage.get("completion_tokens") is not None:
            completion_tokens = int(usage["completion_tokens"])
            saw_completion = True

    timings = parsed_resp.get("timings")
    if isinstance(timings, Mapping):
        if not saw_prompt and timings.get("prompt_n") is not None:
            prompt_tokens = int(timings["prompt_n"])
            saw_prompt = True
        if not saw_completion and timings.get("predicted_n") is not None:
            completion_tokens = int(timings["predicted_n"])
            saw_completion = True

    if not saw_completion:
        output_txt = _extract_choice_output_text(parsed_resp)
        if output_txt:
            completion_tokens = max(1, (len(output_txt) + 3) // 4)

    return prompt_tokens, completion_tokens


def _sse_event_should_commit(payload_bytes: bytes) -> bool:
    """Return True when a complete SSE data event carries output or a terminator.

    Role-only / empty-delta chunks (e.g. `{"choices":[{"delta":{"role":"assistant"}}]}`)
    are buffered so Spot preemption before the first generated token still fails over.
    """
    if not payload_bytes:
        return False
    if payload_bytes == b"[DONE]":
        return True
    try:
        obj = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return True
    if not isinstance(obj, Mapping):
        return True
    if _extract_choice_output_text(obj):
        return True
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                return True
        if not choices and (
            isinstance(obj.get("usage"), Mapping) or obj.get("error") is not None
        ):
            return True
        return False
    return True


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
        if not self.config.enabled:
            self.healthy = False
            self.unhealthy_reason = "disabled"

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
            "healthy": self.config.enabled and self.healthy and not in_cooldown,
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
        auth_type = auth_cfg.get("type")
        if spec is not None:
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(
                    f"Invalid auth.hook spec {spec!r}; expected 'module:ClassName'"
                )
            if auth_type is not None and auth_type != "custom":
                raise ValueError(
                    f"Invalid auth.type {auth_type!r} when auth.hook is configured; "
                    "omit 'type' or set 'custom'"
                )
            misplaced = [k for k in BEARER_AUTH_ONLY_KEYS if k in auth_cfg]
            if misplaced:
                raise ValueError(
                    f"Bearer auth settings {misplaced!r} require auth.type='bearer'"
                )
            resolved_auth = load_hook_from_spec(spec.strip(), auth_cfg.get("kwargs"))
        elif auth_type == "bearer":
            raw_tokens = auth_cfg.get("tokens")
            tokens: List[str] = []
            if raw_tokens is not None:
                if not isinstance(raw_tokens, (list, tuple)) or not all(
                    isinstance(t, str) and t.strip() for t in raw_tokens
                ):
                    raise ValueError("auth.tokens must be a list of non-empty strings")
                tokens.extend(t for t in raw_tokens)
            raw_tokens_env = auth_cfg.get("tokens_env")
            if raw_tokens_env is not None:
                if not isinstance(raw_tokens_env, (list, tuple)) or not all(
                    isinstance(e, str) and e.strip() for e in raw_tokens_env
                ):
                    raise ValueError(
                        "auth.tokens_env must be a list of non-empty environment variable names"
                    )
                import os

                for env_name in raw_tokens_env:
                    val = os.environ.get(env_name.strip(), "").strip()
                    if val:
                        tokens.append(val)
            if "exempt_paths" in auth_cfg and auth_cfg["exempt_paths"] is not None:
                raw_exempt = auth_cfg["exempt_paths"]
                if not isinstance(raw_exempt, (list, tuple)) or not all(
                    isinstance(p, str) and p.strip() for p in raw_exempt
                ):
                    raise ValueError(
                        "auth.exempt_paths must be a list of non-empty path strings"
                    )
                exempt: Sequence[str] = [p.strip() for p in raw_exempt]
            else:
                exempt = ("/health",)
            resolved_auth = BearerTokenAuthHook(tokens, exempt_paths=exempt)
        elif auth_type in (None, "allow_all"):
            misplaced = [k for k in BEARER_AUTH_ONLY_KEYS if k in auth_cfg]
            if misplaced:
                raise ValueError(
                    f"Bearer auth settings {misplaced!r} require auth.type='bearer'"
                )
            resolved_auth = AllowAllAuthHook()
        else:
            raise ValueError(
                f"Invalid auth.type {auth_type!r}; expected one of {VALID_AUTH_TYPES}"
            )

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
        self._metric_models_seen: set[str] = set()
        self._poller_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def start_background_poller(self) -> None:
        with self._poller_lock:
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
        with self._poller_lock:
            self._stop_event.set()
            thread = self._poll_thread
            self._poll_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

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
            if not state.config.enabled:
                continue
            try:
                self.probe_backend(state)
            except Exception as exc:
                with self._lock:
                    state.last_probe_unix = time.time()
                    state.healthy = False
                    state.unhealthy_reason = f"probe_exception: {exc}"

    def metric_model_label(
        self,
        model: str,
        *,
        matched_backend: bool = False,
    ) -> str:
        """Map a client-supplied model name to a bounded Prometheus label."""
        with self._lock:
            is_known = False
            if model and model != "*":
                for b in self._backends:
                    if model in b.config.models or model in b.discovered_models:
                        is_known = True
                        break
            if is_known:
                if (
                    model in self._metric_models_seen
                    or len(self._metric_models_seen) < MAX_METRIC_MODEL_LABELS
                ):
                    self._metric_models_seen.add(model)
                    return model
                return OTHER_MODEL_LABEL
            return OTHER_MODEL_LABEL if matched_backend else UNKNOWN_MODEL_LABEL

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
        _ensure_safe_credential_transport(backend, scheme, host, api_key)
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

    def try_reserve_backend(
        self,
        state: BackendState,
        *,
        now_monotonic: Optional[float] = None,
    ) -> Optional[int]:
        """Atomically recheck backend eligibility and reserve one in-flight slot."""
        now_m = now_monotonic if now_monotonic is not None else time.monotonic()
        with self._lock:
            if not state.is_eligible(now_m):
                return None
            state.inflight += 1
            state.total_requests += 1
            state.last_used_monotonic = now_m
            state.idle_notified = False
        return self.metrics.inc_inflight(state.config.id)

    def release_backend(self, state: BackendState) -> None:
        """Release one in-flight slot; start idle timing when in-flight drops to zero."""
        now_m = time.monotonic()
        with self._lock:
            state.inflight = max(0, state.inflight - 1)
            if state.inflight == 0:
                state.last_used_monotonic = now_m
                state.idle_notified = False
        self.metrics.dec_inflight(state.config.id)

    def select_candidates(
        self,
        model: str,
        *,
        policy: Optional[str] = None,
        exclude_ids: Optional[Sequence[str] | set[str]] = None,
    ) -> List[BackendState]:
        active_policy = policy or self.config.policy
        if active_policy not in VALID_POLICIES:
            active_policy = self.config.policy
        excluded = set(exclude_ids) if exclude_ids else set()

        now_m = time.monotonic()
        with self._lock:
            eligible = [
                b
                for b in self._backends
                if b.config.id not in excluded
                and b.is_eligible(now_m)
                and b.supports_model(model)
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
            rr_key = self.metric_model_label(model, matched_backend=True)
            cursor = self._rr_cursors.get(rr_key, 0)
            if not excluded:
                self._rr_cursors[rr_key] = (cursor + 1) % len(eligible)
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
        healthy_count = sum(1 for s in snaps if s["enabled"] and s["healthy"])
        enabled_count = sum(1 for s in snaps if s["enabled"])
        status_str = "ok" if healthy_count > 0 else "degraded"
        http_code = 200 if healthy_count > 0 else 503
        payload = {
            "status": status_str,
            "version": __version__,
            "policy": self.config.policy,
            "healthy_backends": healthy_count,
            "enabled_backends": enabled_count,
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
        metric_model = self.metric_model_label(model, matched_backend=True)
        self.metrics.record_request(
            model=metric_model,
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


class _DeadlineReader:
    """Buffered reader wrapper enforcing a strict wall-clock deadline on header lines."""

    def __init__(self, raw_rfile: Any, sock: Any, deadline: float) -> None:
        self._raw = raw_rfile
        self._sock = sock
        self._deadline = deadline
        self._buf = bytearray()

    def readline(self, limit: int = -1) -> bytes:
        max_len = limit if limit is not None and limit >= 0 else None
        read_fn = getattr(self._raw, "read1", self._raw.read)
        while True:
            nl_idx = self._buf.find(b"\n")
            if nl_idx != -1:
                end = nl_idx + 1
                if max_len is not None and end > max_len:
                    end = max_len
                out = bytes(self._buf[:end])
                del self._buf[:end]
                return out
            if max_len is not None and len(self._buf) >= max_len:
                out = bytes(self._buf[:max_len])
                del self._buf[:max_len]
                return out
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Client header read timed out")
            self._sock.settimeout(remaining)
            chunk = read_fn(4096)
            if not chunk:
                out = bytes(self._buf)
                self._buf.clear()
                return out
            self._buf.extend(chunk)

    def read1(self, n: int = -1) -> bytes:
        if self._buf:
            if n is None or n < 0:
                out = bytes(self._buf)
                self._buf.clear()
                return out
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out
        read_fn = getattr(self._raw, "read1", self._raw.read)
        return read_fn(n)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            prefix = bytes(self._buf)
            self._buf.clear()
            return prefix + self._raw.read()
        out = bytearray(self._buf[:n])
        del self._buf[: len(out)]
        while len(out) < n:
            chunk = self._raw.read(n - len(out))
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

    def close(self) -> None:
        self._raw.close()


_OVERLOAD_503_BODY = _openai_error_body(
    "Server connection limit reached; retry later",
    error_type="service_unavailable",
    code="server_overloaded",
)
_OVERLOAD_503_RAW = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: application/json\r\n"
    + f"Content-Length: {len(_OVERLOAD_503_BODY)}\r\n".encode("ascii")
    + b"Connection: close\r\n\r\n"
    + _OVERLOAD_503_BODY
)


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
        self._conn_semaphore = threading.BoundedSemaphore(
            engine.config.max_concurrent_connections
        )
        super().__init__(server_address, _RouterRequestHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._conn_semaphore.acquire(blocking=False):
            try:
                request.setblocking(False)
                request.send(_OVERLOAD_503_RAW)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._conn_semaphore.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._conn_semaphore.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        _, exc_val, _ = sys.exc_info()
        if isinstance(exc_val, (BrokenPipeError, ConnectionResetError)) and not self.verbose:
            return
        super().handle_error(request, client_address)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self.engine.start_background_poller()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.engine.stop_background_poller()

    def server_close(self) -> None:
        self.engine.stop_background_poller()
        super().server_close()


class _RouterRequestHandler(BaseHTTPRequestHandler):
    server: FreeTPURouterServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.timeout = self.server.engine.config.client_header_timeout_s
        super().setup()
        header_deadline = (
            time.monotonic() + self.server.engine.config.client_header_timeout_s
        )
        self.rfile = _DeadlineReader(self.rfile, self.connection, header_deadline)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _resolve_cors_header(self) -> Tuple[bool, Optional[str]]:
        """Check request Origin against explicit `cors_allowed_origins`.

        Returns `(allowed, acao_header_value)`. Non-browser requests without an
        `Origin` header are allowed with `acao_header_value=None`.
        """
        raw_origin = self.headers.get("Origin")
        if raw_origin is None:
            return True, None
        origin = raw_origin.strip()
        allowed = self.server.engine.config.cors_allowed_origins
        if not allowed:
            return False, None
        if "*" in allowed:
            return True, "*"
        if origin.rstrip("/") in allowed:
            return True, origin
        return False, None

    def _send_cors_headers(self, acao: Optional[str]) -> None:
        if acao is None:
            return
        self.send_header("Access-Control-Allow-Origin", acao)
        if acao != "*":
            self.send_header("Vary", "Origin")

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.close_connection = True
        _, acao = self._resolve_cors_header()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers(acao)
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.wfile.flush()

    def _safe_send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        try:
            self._send_bytes(status, content_type, body, extra_headers=extra_headers)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True

    def _send_json(
        self,
        status: int,
        payload: Any,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self._send_bytes(status, "application/json", body, extra_headers=extra_headers)

    def _read_request_body(self, content_len: int) -> Tuple[Optional[bytes], Optional[Tuple[int, str, str]]]:
        """Read exactly `content_len` bytes within the configured wall-clock deadline."""
        timeout_s = self.server.engine.config.client_read_timeout_s
        deadline = time.monotonic() + timeout_s
        chunks: List[bytes] = []
        remaining = content_len
        read_fn = getattr(self.rfile, "read1", self.rfile.read)
        try:
            while remaining > 0:
                time_left = deadline - time.monotonic()
                if time_left <= 0:
                    return None, (408, "request_timeout", "Request body read timed out")
                self.connection.settimeout(time_left)
                chunk = read_fn(min(65536, remaining))
                if not chunk:
                    return None, (400, "incomplete_body", "Incomplete request body")
                chunks.append(chunk)
                remaining -= len(chunk)
        except TimeoutError:
            return None, (408, "request_timeout", "Request body read timed out")
        except OSError:
            return None, (400, "incomplete_body", "Client disconnected while uploading body")
        return b"".join(chunks), None

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
        self.close_connection = True
        allowed, acao = self._resolve_cors_header()
        if not allowed or acao is None:
            self._send_bytes(
                403,
                "application/json",
                _openai_error_body(
                    "Cross-origin requests are not allowed by router CORS policy",
                    error_type="permission_error",
                    code="origin_not_allowed",
                ),
            )
            return
        self.send_response(204)
        self._send_cors_headers(acao)
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
        origin_allowed, _ = self._resolve_cors_header()
        if not origin_allowed:
            self._send_bytes(
                403,
                "application/json",
                _openai_error_body(
                    "Origin is not allowed by router CORS policy",
                    error_type="permission_error",
                    code="origin_not_allowed",
                ),
            )
            return

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

        max_body = self.server.engine.config.max_request_body_bytes
        if content_len > max_body:
            self._send_bytes(
                413,
                "application/json",
                _openai_error_body(
                    f"Request body exceeds maximum size of {max_body} bytes",
                    code="payload_too_large",
                ),
            )
            return

        raw_req, read_err = self._read_request_body(content_len)
        if read_err is not None or raw_req is None:
            err_status, err_code, err_msg = read_err or (
                400,
                "incomplete_body",
                "Failed to read request body",
            )
            self._safe_send_bytes(
                err_status,
                "application/json",
                _openai_error_body(err_msg, code=err_code),
            )
            return

        # Allow legitimate long-running inference/streaming responses to write cleanly.
        try:
            write_timeout = max(
                self.server.engine.config.client_read_timeout_s,
                self.server.engine.config.request_timeout_s,
            )
            self.connection.settimeout(write_timeout)
        except OSError:
            pass

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

        est_prompt_tokens = _estimate_prompt_tokens(req_payload)
        failovers = 0
        last_error_status = 503
        last_error_msg = f"No healthy backend available for model {model!r}"
        last_error_type = "service_unavailable"
        last_error_code = "no_healthy_backend"
        last_error_body: Optional[bytes] = None
        last_error_content_type = "application/json"
        last_attempted_backend: Optional[BackendState] = None
        prev_failure: Optional[Tuple[str, str]] = None
        pressure_checked = False
        tried_ids: set[str] = set()

        while True:
            candidates = engine.select_candidates(
                model,
                policy=policy_override,
                exclude_ids=tried_ids,
            )
            backend: Optional[BackendState] = None
            total_inflight_now: Optional[int] = None
            attempt_start = time.monotonic()
            for cand in candidates:
                if cand.config.id in tried_ids:
                    continue
                attempt_start = time.monotonic()
                reserved = engine.try_reserve_backend(
                    cand, now_monotonic=attempt_start
                )
                if reserved is not None:
                    backend = cand
                    total_inflight_now = reserved
                    break
            if backend is None or total_inflight_now is None:
                break

            tried_ids.add(backend.config.id)

            if not pressure_checked:
                pressure_checked = True
                if total_inflight_now >= engine.config.queue_pressure_threshold:
                    engine._emit_queue_pressure(
                        model=model,
                        active_inflight=total_inflight_now,
                        healthy_backends=len(candidates),
                        reason="inflight_threshold_exceeded",
                    )

            if prev_failure is not None:
                failovers += 1
                engine.metrics.record_failover(
                    model=engine.metric_model_label(model, matched_backend=True),
                    from_backend=prev_failure[0],
                    to_backend=backend.config.id,
                    reason=prev_failure[1],
                )
                prev_failure = None

            last_attempted_backend = backend
            conn: Optional[http.client.HTTPConnection] = None
            resp: Optional[http.client.HTTPResponse] = None
            non_stream_result: Optional[Tuple[int, str, bytes, float, float]] = None
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
                _ensure_safe_credential_transport(backend.config, scheme, host, api_key)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                conn.request("POST", upstream_path, body=raw_req, headers=headers)
                upstream_sock = conn.sock
                resp = conn.getresponse()
                if upstream_sock is None:
                    upstream_sock = getattr(
                        getattr(resp.fp, "raw", None), "_sock", None
                    )

                if resp.status in RETRYABLE_STATUS_CODES:
                    # Fail over immediately without waiting to drain a potentially stalled error body.
                    reason = f"http_{resp.status}"
                    last_error_status = resp.status
                    last_error_msg = (
                        f"Backend {backend.config.id!r} returned {resp.status}"
                    )
                    last_error_type = "service_unavailable"
                    last_error_code = "backend_unavailable"
                    last_error_body = None
                    last_error_content_type = "application/json"
                    engine.mark_backend_failed(backend, reason)
                    prev_failure = (backend.config.id, reason)
                    continue

                if stream and 200 <= resp.status < 300:
                    completed = self._proxy_sse_stream(
                        resp=resp,
                        upstream_sock=upstream_sock,
                        timeout_s=timeout_s,
                        backend=backend,
                        model=model,
                        expected_choices=_expected_stream_choices(endpoint, req_payload),
                        req_id=req_id,
                        req_start=req_start,
                        attempt_start=attempt_start,
                        failovers=failovers,
                        est_prompt_tokens=est_prompt_tokens,
                        principal=auth_ctx.principal,
                    )
                    if completed:
                        return
                    # Pre-first-token stream failure: fail over to next candidate.
                    reason = "stream_aborted_before_first_token"
                    last_error_status = 502
                    last_error_msg = (
                        f"Backend {backend.config.id!r} aborted stream before first token"
                    )
                    last_error_type = "bad_gateway"
                    last_error_code = "upstream_stream_error"
                    last_error_body = None
                    last_error_content_type = "application/json"
                    engine.mark_backend_failed(backend, reason)
                    prev_failure = (backend.config.id, reason)
                    continue

                # Non-streaming response (or non-retryable error status such as 4xx)
                raw_resp = resp.read()
                latency_ms = (time.monotonic() - req_start) * 1000.0
                attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                content_type = resp.getheader("Content-Type") or "application/json"
                non_stream_result = (
                    resp.status,
                    content_type,
                    raw_resp,
                    latency_ms,
                    attempt_ms,
                )
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                reason = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
                last_error_status = 504 if isinstance(exc, TimeoutError) else 502
                last_error_msg = f"Backend {backend.config.id!r} failed: {exc}"
                last_error_type = "service_unavailable"
                last_error_code = reason
                last_error_body = None
                last_error_content_type = "application/json"
                engine.mark_backend_failed(backend, reason)
                prev_failure = (backend.config.id, reason)
                continue
            finally:
                engine.release_backend(backend)
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

            if non_stream_result is not None:
                resp_status, content_type, raw_resp, latency_ms, attempt_ms = (
                    non_stream_result
                )
                if 200 <= resp_status < 300:
                    with engine._lock:
                        backend.record_latency(attempt_ms)
                    prompt_tokens, completion_tokens = _extract_non_stream_usage(
                        raw_resp, est_prompt_tokens
                    )
                else:
                    prompt_tokens = 0
                    completion_tokens = 0

                engine._record_usage(
                    request_id=req_id,
                    model=model,
                    backend=backend,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    ttft_ms=latency_ms if 200 <= resp_status < 300 else None,
                    stream=False,
                    failovers=failovers,
                    principal=auth_ctx.principal,
                    status_code=resp_status,
                )
                # Keep downstream client write failures isolated from backend failover.
                self._safe_send_bytes(
                    resp_status,
                    content_type,
                    raw_resp,
                    extra_headers={
                        "X-FreeTPU-Backend": backend.config.id,
                        "X-FreeTPU-Failovers": str(failovers),
                        "X-FreeTPU-Request-Id": req_id,
                    },
                )
                return

        total_latency_ms = (time.monotonic() - req_start) * 1000.0
        if last_attempted_backend is None:
            engine._emit_queue_pressure(
                model=model,
                active_inflight=engine.metrics.total_inflight(),
                healthy_backends=0,
                reason="no_healthy_backend",
            )
            engine.metrics.record_request(
                model=engine.metric_model_label(model, matched_backend=False),
                backend_id="none",
                status=503,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=total_latency_ms,
                ttft_ms=None,
                estimated_cost_usd=0.0,
            )
            self._safe_send_bytes(
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

        engine.metrics.record_request(
            model=engine.metric_model_label(model, matched_backend=True),
            backend_id=last_attempted_backend.config.id,
            status=last_error_status,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=total_latency_ms,
            ttft_ms=None,
            estimated_cost_usd=0.0,
        )
        self._safe_send_bytes(
            last_error_status,
            last_error_content_type,
            last_error_body
            or _openai_error_body(
                last_error_msg,
                error_type=last_error_type,
                code=last_error_code,
            ),
            extra_headers={
                "X-FreeTPU-Backend": last_attempted_backend.config.id,
                "X-FreeTPU-Failovers": str(failovers),
                "X-FreeTPU-Request-Id": req_id,
            },
        )

    def _proxy_sse_stream(
        self,
        *,
        resp: http.client.HTTPResponse,
        upstream_sock: Any,
        timeout_s: float,
        backend: BackendState,
        model: str,
        expected_choices: int,
        req_id: str,
        req_start: float,
        attempt_start: float,
        failovers: int,
        est_prompt_tokens: int,
        principal: str,
    ) -> bool:
        """Buffer until first generated output or terminator before committing 200 OK.

        Uses incremental bounded reads via `_BoundedSSEReader` during pre-output
        buffering so partial lines without newlines cannot exceed the byte cap or
        stall past the elapsed deadline. Uses HTTP/1.1 chunked transfer encoding
        for the committed SSE response so premature upstream disconnects omit the
        terminating `0\\r\\n\\r\\n` chunk (in addition to emitting an SSE error event)
        and are surfaced to HTTP clients as incomplete streams rather than clean
        completions.

        Returns True if headers/stream were committed to the downstream client,
        or False if upstream closed or sent no qualifying output before commit.
        """
        engine = self.server.engine
        sse_reader = _BoundedSSEReader(resp, upstream_sock)
        pre_commit_lines: List[bytes] = []
        pre_commit_bytes = 0
        pre_commit_events = 0
        max_pre_commit_bytes = engine.config.max_sse_pre_commit_bytes
        max_pre_commit_events = 64
        first_chunk_time: Optional[float] = None
        pending_event_data: List[bytes] = []
        pre_commit_deadline = attempt_start + timeout_s

        # Phase 1: upstream-only incremental read through HTTPResponse.read1().
        # Enforce both byte cap and elapsed deadline as bytes arrive, even without newlines.
        while True:
            remaining_bytes = max_pre_commit_bytes - pre_commit_bytes
            if remaining_bytes <= 0:
                return False
            line, exceeded_cap = sse_reader.readline(
                deadline=pre_commit_deadline,
                max_bytes=remaining_bytes,
            )
            if exceeded_cap or not line:
                # Byte cap exceeded or EOF before generated output/terminator -> retryable failure.
                return False
            pre_commit_bytes += len(line)
            pre_commit_lines.append(line)
            if line in (b"\n", b"\r\n", b"\r"):
                if pending_event_data:
                    pre_commit_events += 1
                    payload_bytes = b"\n".join(pending_event_data).strip()
                    pending_event_data.clear()
                    if _sse_event_should_commit(payload_bytes):
                        first_chunk_time = time.monotonic()
                        break
                    if pre_commit_events >= max_pre_commit_events:
                        return False
                continue
            content_line = line.rstrip(b"\r\n")
            if content_line.startswith(b":"):
                continue
            if content_line.startswith(b"data:"):
                data_val = content_line[5:]
                if data_val.startswith(b" "):
                    data_val = data_val[1:]
                pending_event_data.append(data_val)

        # Restore per-read upstream socket timeout for post-commit streaming.
        if upstream_sock is not None:
            try:
                upstream_sock.settimeout(timeout_s)
            except OSError:
                pass

        # Phase 2: upstream delivered output or valid terminator -> commit chunked response.
        # Downstream write errors from here on must never trigger backend failover.
        ttft_ms: Optional[float] = (
            (first_chunk_time - req_start) * 1000.0 if first_chunk_time else None
        )
        first_content_ttft_ms: Optional[float] = None
        prompt_tokens = est_prompt_tokens
        completion_tokens = 0
        est_completion_chars = 0
        saw_explicit_prompt_usage = False
        saw_explicit_completion_usage = False
        saw_done_sentinel = False
        finished_choice_indices: set[int] = set()
        upstream_read_error = False
        downstream_write_error = False
        event_data_lines: List[bytes] = []
        required_choices = max(1, expected_choices)

        def process_sse_event() -> None:
            nonlocal first_content_ttft_ms, prompt_tokens, completion_tokens
            nonlocal est_completion_chars, saw_explicit_prompt_usage
            nonlocal saw_explicit_completion_usage, saw_done_sentinel
            if not event_data_lines:
                return
            payload_bytes = b"\n".join(event_data_lines).strip()
            event_data_lines.clear()
            if not payload_bytes:
                return
            if payload_bytes == b"[DONE]":
                saw_done_sentinel = True
                return
            try:
                obj = json.loads(payload_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                return
            if not isinstance(obj, Mapping):
                return
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                        c_idx = choice.get("index")
                        if isinstance(c_idx, int) and not isinstance(c_idx, bool) and c_idx >= 0:
                            finished_choice_indices.add(c_idx)
                        else:
                            finished_choice_indices.add(len(finished_choice_indices))
            output_txt = _extract_choice_output_text(obj)
            if output_txt:
                if first_content_ttft_ms is None:
                    first_content_ttft_ms = (time.monotonic() - req_start) * 1000.0
                est_completion_chars += len(output_txt)
            usage = obj.get("usage")
            if isinstance(usage, Mapping):
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = int(usage["prompt_tokens"])
                    saw_explicit_prompt_usage = True
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                    saw_explicit_completion_usage = True
            timings = obj.get("timings")
            if isinstance(timings, Mapping):
                if not saw_explicit_prompt_usage and timings.get("prompt_n") is not None:
                    prompt_tokens = int(timings["prompt_n"])
                    saw_explicit_prompt_usage = True
                if (
                    not saw_explicit_completion_usage
                    and timings.get("predicted_n") is not None
                ):
                    completion_tokens = int(timings["predicted_n"])
                    saw_explicit_completion_usage = True

        def inspect_sse_line(raw_line: bytes) -> None:
            if raw_line in (b"\n", b"\r\n", b"\r"):
                process_sse_event()
                return
            content_line = raw_line.rstrip(b"\r\n")
            if content_line.startswith(b"data:"):
                data_val = content_line[5:]
                if data_val.startswith(b" "):
                    data_val = data_val[1:]
                event_data_lines.append(data_val)

        def write_chunk(data: bytes) -> None:
            if not data:
                return
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")

        self.close_connection = True
        _, acao = self._resolve_cors_header()
        try:
            self.send_response(resp.status)
            self.send_header(
                "Content-Type",
                resp.getheader("Content-Type") or "text/event-stream; charset=utf-8",
            )
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self._send_cors_headers(acao)
            self.send_header("Connection", "close")
            self.send_header("X-FreeTPU-Backend", backend.config.id)
            self.send_header("X-FreeTPU-Failovers", str(failovers))
            self.send_header("X-FreeTPU-Request-Id", req_id)
            self.end_headers()

            for buffered in pre_commit_lines:
                inspect_sse_line(buffered)
            write_chunk(b"".join(pre_commit_lines))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            downstream_write_error = True

        if not downstream_write_error:
            event_chunk_buf = bytearray()
            while True:
                try:
                    line, _ = sse_reader.readline()
                except (OSError, TimeoutError, http.client.HTTPException):
                    upstream_read_error = True
                    break
                if not line:
                    break
                inspect_sse_line(line)
                event_chunk_buf.extend(line)
                if line in (b"\n", b"\r\n", b"\r"):
                    try:
                        write_chunk(bytes(event_chunk_buf))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        downstream_write_error = True
                        break
                    event_chunk_buf.clear()

            if event_chunk_buf and not downstream_write_error:
                try:
                    write_chunk(bytes(event_chunk_buf))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    downstream_write_error = True

        process_sse_event()
        all_choices_finished = all(
            idx in finished_choice_indices for idx in range(required_choices)
        )
        stream_completed = saw_done_sentinel or all_choices_finished
        premature_upstream_end = (
            not downstream_write_error
            and (upstream_read_error or not stream_completed)
        )
        if not saw_explicit_completion_usage and est_completion_chars > 0:
            completion_tokens = max(1, (est_completion_chars + 3) // 4)
        total_ms = (time.monotonic() - req_start) * 1000.0
        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
        effective_ttft = (
            first_content_ttft_ms if first_content_ttft_ms is not None else ttft_ms
        )
        if premature_upstream_end:
            engine.mark_backend_failed(backend, "stream_truncated_mid_response")
            recorded_status = 502
        else:
            with engine._lock:
                backend.record_latency(attempt_ms)
            recorded_status = resp.status

        # Record metrics and usage before sending the final 0-chunk so callers
        # inspecting /metrics or UsageMeterHook immediately after stream EOF see it.
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
            status_code=recorded_status,
        )

        if not downstream_write_error:
            try:
                if premature_upstream_end:
                    # Emit SSE error event and intentionally omit the terminating 0-chunk so
                    # HTTP/1.1 chunked clients surface premature truncation instead of clean EOF.
                    err_event = (
                        b"event: error\n"
                        b'data: {"error":{"message":"Upstream stream terminated before completion",'
                        b'"type":"server_error","code":"upstream_stream_interrupted"}}\n\n'
                    )
                    write_chunk(err_event)
                    self.wfile.flush()
                else:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                downstream_write_error = True

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
        "--cors-origin",
        action="append",
        dest="cors_origins",
        help=(
            "Explicitly allow a browser Origin for CORS (repeatable; cross-origin "
            "access is disabled by default)."
        ),
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

    try:
        cfg = load_config(args.config)
        if args.host is not None:
            cfg.host = args.host
        if args.port is not None:
            cfg.port = args.port
        if args.policy is not None:
            cfg.policy = args.policy
        if args.bearer_token:
            cfg.auth = {"type": "bearer", "tokens": [args.bearer_token]}
        if args.cors_origins:
            cfg.cors_allowed_origins = list(args.cors_origins)
        if args.usage_jsonl:
            cfg.usage_meter = {"type": "jsonl", "jsonl_path": str(args.usage_jsonl)}
        cfg.validate()
        if args.check_config:
            build_hooks_from_config(cfg)
    except (OSError, ValueError, json.JSONDecodeError, ImportError, AttributeError) as exc:
        sys.stderr.write(f"Config error: {exc}\n")
        return 1

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
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
