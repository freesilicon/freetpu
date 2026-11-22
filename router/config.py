#!/usr/bin/env python3
"""Configuration schema and JSON loader for FreeTPU Router v0."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

VALID_POLICIES = ("cheapest", "local_first", "lowest_latency", "round_robin")
VALID_AUTH_TYPES = ("allow_all", "bearer")
VALID_URL_SCHEMES = ("http", "https")
BEARER_AUTH_ONLY_KEYS = ("tokens", "tokens_env", "exempt_paths")
DEFAULT_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
DEFAULT_CLIENT_READ_TIMEOUT_S = 30.0
DEFAULT_CLIENT_HEADER_TIMEOUT_S = 10.0
DEFAULT_MAX_CONCURRENT_CONNECTIONS = 128
DEFAULT_MAX_SSE_PRE_COMMIT_BYTES = 1024 * 1024


def _require_bool(
    raw: Mapping[str, Any],
    key: str,
    default: bool,
    *,
    context: str,
) -> bool:
    if key not in raw:
        return default
    val = raw[key]
    if not isinstance(val, bool):
        raise ValueError(
            f"{context} field {key!r} must be a JSON boolean (true/false), got {val!r}"
        )
    return val


def _validate_finite_number(
    val: Any,
    *,
    name: str,
    min_value: Optional[float] = None,
    strictly_greater: bool = False,
    max_value: Optional[float] = None,
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if min_value is not None:
        if strictly_greater and fval <= min_value:
            raise ValueError(f"{name} must be > {min_value}, got {val!r}")
        if not strictly_greater and fval < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {val!r}")
    if max_value is not None and fval > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {val!r}")
    return fval


def _validate_int_range(
    val: Any,
    *,
    name: str,
    min_value: int,
    max_value: Optional[int] = None,
) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer, got {val!r}")
    if val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {val!r}")
    if max_value is not None and val > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {val!r}")
    return val


def is_loopback_host(host: str) -> bool:
    """Return True if `host` is a loopback hostname (`localhost`) or IP (`127.0.0.0/8`, `::1`)."""
    cleaned = host.strip().lower().rstrip(".")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned == "localhost":
        return True
    addr_str = cleaned.split("%", 1)[0]
    try:
        return ipaddress.ip_address(addr_str).is_loopback
    except ValueError:
        return False


def sanitize_url(url: str) -> str:
    """Strip any userinfo credentials from a URL before logging or displaying."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    _, hostport = parts.netloc.rsplit("@", 1)
    return parts._replace(netloc=hostport).geturl()


def validate_http_url(url: str, *, field_name: str = "url") -> str:
    """Validate that a backend or exporter URL uses an explicit http/https scheme, host, and port."""
    cleaned = url.strip()
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in VALID_URL_SCHEMES:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: "
            f"scheme must be one of {VALID_URL_SCHEMES}"
        )
    if not parts.hostname:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: missing hostname"
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: invalid port ({exc})"
        ) from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: "
            f"port {port} out of range 1..65535"
        )
    return cleaned


@dataclass
class BackendConfig:
    """Configuration for a single heterogeneous inference backend."""

    id: str
    base_url: str
    type: str = "generic"
    local: bool = False
    spot: bool = False
    models: List[str] = field(default_factory=list)
    discover_models: bool = True
    cost_per_1m_prompt_usd: float = 0.0
    cost_per_1m_completion_usd: float = 0.0
    priority: int = 100
    weight: int = 1
    max_inflight: Optional[int] = None
    timeout_s: Optional[float] = None
    health_path: Optional[str] = None
    tpu_exporter_url: Optional[str] = None
    max_hbm_utilization: float = 0.95
    max_duty_cycle_percent: Optional[float] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    allow_insecure_http: bool = False
    latency_hint_ms: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Backend 'id' must be non-empty")
        for field_name in (
            "local",
            "spot",
            "discover_models",
            "allow_insecure_http",
            "enabled",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise ValueError(
                    f"Backend {self.id!r} field {field_name!r} must be a boolean, got {val!r}"
                )
        self.base_url = validate_http_url(
            self.base_url, field_name=f"backend {self.id!r} base_url"
        ).rstrip("/")
        if self.tpu_exporter_url:
            self.tpu_exporter_url = validate_http_url(
                self.tpu_exporter_url,
                field_name=f"backend {self.id!r} tpu_exporter_url",
            )
        ctx = f"Backend {self.id!r}"
        _validate_finite_number(
            self.cost_per_1m_prompt_usd,
            name=f"{ctx} cost_per_1m_prompt_usd",
            min_value=0.0,
        )
        _validate_finite_number(
            self.cost_per_1m_completion_usd,
            name=f"{ctx} cost_per_1m_completion_usd",
            min_value=0.0,
        )
        _validate_finite_number(
            self.max_hbm_utilization,
            name=f"{ctx} max_hbm_utilization",
            min_value=0.0,
            strictly_greater=True,
            max_value=1.0,
        )
        if self.max_duty_cycle_percent is not None:
            _validate_finite_number(
                self.max_duty_cycle_percent,
                name=f"{ctx} max_duty_cycle_percent",
                min_value=0.0,
                strictly_greater=True,
                max_value=100.0,
            )
        if self.max_inflight is not None:
            _validate_int_range(
                self.max_inflight, name=f"{ctx} max_inflight", min_value=1
            )
        if self.timeout_s is not None:
            _validate_finite_number(
                self.timeout_s,
                name=f"{ctx} timeout_s",
                min_value=0.0,
                strictly_greater=True,
            )
        if self.latency_hint_ms is not None:
            _validate_finite_number(
                self.latency_hint_ms,
                name=f"{ctx} latency_hint_ms",
                min_value=0.0,
            )
        _validate_int_range(self.priority, name=f"{ctx} priority", min_value=-1_000_000)
        _validate_int_range(self.weight, name=f"{ctx} weight", min_value=1)
        has_credentials = bool(
            (self.api_key and self.api_key.strip())
            or (self.api_key_env and self.api_key_env.strip())
        )
        parts = urlsplit(self.base_url)
        if (
            has_credentials
            and parts.scheme.lower() == "http"
            and not is_loopback_host(parts.hostname or "")
            and self.allow_insecure_http is not True
        ):
            raise ValueError(
                f"Backend {self.id!r} configures bearer credentials over plaintext HTTP "
                f"to non-loopback host {parts.hostname!r}; use 'https://' or explicitly "
                "set 'allow_insecure_http': true"
            )

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            val = os.environ.get(self.api_key_env, "").strip()
            return val or None
        return None

    def blended_cost_per_1m_tokens(self) -> float:
        return (self.cost_per_1m_prompt_usd + self.cost_per_1m_completion_usd) / 2.0

    def estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (max(0, prompt_tokens) * self.cost_per_1m_prompt_usd)
            + (max(0, completion_tokens) * self.cost_per_1m_completion_usd)
        ) / 1_000_000.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BackendConfig":
        backend_id = str(raw["id"]).strip()
        if not backend_id:
            raise ValueError("Backend 'id' must be non-empty")
        base_url = str(raw["base_url"]).strip().rstrip("/")
        if not base_url:
            raise ValueError(f"Backend {backend_id!r} requires a non-empty 'base_url'")
        btype = str(raw.get("type", "generic")).strip() or "generic"
        ctx = f"Backend {backend_id!r}"
        if "local" in raw:
            is_local = _require_bool(raw, "local", False, context=ctx)
        else:
            lowered = btype.lower()
            is_local = lowered.startswith(("pixel", "local", "mac", "edge"))

        models_raw = raw.get("models") or []
        models = [str(m).strip() for m in models_raw if str(m).strip()]

        cost_shared = raw.get("cost_per_1m_tokens_usd")
        prompt_cost = _validate_finite_number(
            raw.get(
                "cost_per_1m_prompt_usd",
                cost_shared if cost_shared is not None else 0.0,
            ),
            name=f"{ctx} cost_per_1m_prompt_usd",
            min_value=0.0,
        )
        completion_cost = _validate_finite_number(
            raw.get(
                "cost_per_1m_completion_usd",
                cost_shared if cost_shared is not None else 0.0,
            ),
            name=f"{ctx} cost_per_1m_completion_usd",
            min_value=0.0,
        )

        max_hbm = _validate_finite_number(
            raw.get("max_hbm_utilization", 0.95),
            name=f"{ctx} max_hbm_utilization",
            min_value=0.0,
            strictly_greater=True,
            max_value=1.0,
        )

        max_duty = raw.get("max_duty_cycle_percent")
        max_duty_val = (
            _validate_finite_number(
                max_duty,
                name=f"{ctx} max_duty_cycle_percent",
                min_value=0.0,
                strictly_greater=True,
                max_value=100.0,
            )
            if max_duty is not None
            else None
        )

        max_inf = raw.get("max_inflight")
        max_inf_val = (
            _validate_int_range(max_inf, name=f"{ctx} max_inflight", min_value=1)
            if max_inf is not None
            else None
        )

        timeout_raw = raw.get("timeout_s")
        timeout_val: Optional[float] = None
        if timeout_raw is not None:
            timeout_val = _validate_finite_number(
                timeout_raw,
                name=f"{ctx} timeout_s",
                min_value=0.0,
                strictly_greater=True,
            )

        lat_hint_raw = raw.get("latency_hint_ms")
        lat_hint_val: Optional[float] = None
        if lat_hint_raw is not None:
            lat_hint_val = _validate_finite_number(
                lat_hint_raw,
                name=f"{ctx} latency_hint_ms",
                min_value=0.0,
            )

        priority_val = _validate_int_range(
            raw.get("priority", 100), name=f"{ctx} priority", min_value=-1_000_000
        )
        weight_val = _validate_int_range(
            raw.get("weight", 1), name=f"{ctx} weight", min_value=1
        )

        return cls(
            id=backend_id,
            base_url=base_url,
            type=btype,
            local=is_local,
            spot=_require_bool(raw, "spot", False, context=ctx),
            models=models,
            discover_models=_require_bool(raw, "discover_models", True, context=ctx),
            cost_per_1m_prompt_usd=prompt_cost,
            cost_per_1m_completion_usd=completion_cost,
            priority=priority_val,
            weight=weight_val,
            max_inflight=max_inf_val,
            timeout_s=timeout_val,
            health_path=(
                str(raw["health_path"]).strip() if raw.get("health_path") else None
            ),
            tpu_exporter_url=(
                str(raw["tpu_exporter_url"]).strip()
                if raw.get("tpu_exporter_url")
                else None
            ),
            max_hbm_utilization=max_hbm,
            max_duty_cycle_percent=max_duty_val,
            api_key=str(raw["api_key"]) if raw.get("api_key") else None,
            api_key_env=(
                str(raw["api_key_env"]).strip() if raw.get("api_key_env") else None
            ),
            allow_insecure_http=_require_bool(
                raw, "allow_insecure_http", False, context=ctx
            ),
            latency_hint_ms=lat_hint_val,
            enabled=_require_bool(raw, "enabled", True, context=ctx),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass
class RouterConfig:
    """Top-level FreeTPU Router configuration."""

    backends: List[BackendConfig]
    host: str = "127.0.0.1"
    port: int = 8080
    policy: str = "local_first"
    health_poll_interval_s: float = 10.0
    failure_cooldown_s: float = 15.0
    request_timeout_s: float = 120.0
    probe_timeout_s: float = 3.0
    queue_pressure_threshold: int = 16
    backend_idle_timeout_s: float = 300.0
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES
    client_read_timeout_s: float = DEFAULT_CLIENT_READ_TIMEOUT_S
    client_header_timeout_s: float = DEFAULT_CLIENT_HEADER_TIMEOUT_S
    max_concurrent_connections: int = DEFAULT_MAX_CONCURRENT_CONNECTIONS
    max_sse_pre_commit_bytes: int = DEFAULT_MAX_SSE_PRE_COMMIT_BYTES
    cors_allowed_origins: List[str] = field(default_factory=list)
    auth: Dict[str, Any] = field(default_factory=dict)
    usage_meter: Dict[str, Any] = field(default_factory=dict)
    scaler: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.policy not in VALID_POLICIES:
            raise ValueError(
                f"Invalid routing policy {self.policy!r}; expected one of {VALID_POLICIES}"
            )
        seen_ids: set[str] = set()
        for b in self.backends:
            b.validate()
            if b.id in seen_ids:
                raise ValueError(f"Duplicate backend id {b.id!r}")
            seen_ids.add(b.id)
        _validate_int_range(self.port, name="port", min_value=0, max_value=65535)
        _validate_finite_number(
            self.health_poll_interval_s,
            name="health_poll_interval_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_finite_number(
            self.failure_cooldown_s,
            name="failure_cooldown_s",
            min_value=0.0,
        )
        _validate_finite_number(
            self.request_timeout_s,
            name="request_timeout_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_finite_number(
            self.probe_timeout_s,
            name="probe_timeout_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_int_range(
            self.queue_pressure_threshold,
            name="queue_pressure_threshold",
            min_value=1,
        )
        _validate_finite_number(
            self.backend_idle_timeout_s,
            name="backend_idle_timeout_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_int_range(
            self.max_request_body_bytes,
            name="max_request_body_bytes",
            min_value=1,
        )
        _validate_int_range(
            self.max_response_body_bytes,
            name="max_response_body_bytes",
            min_value=1,
        )
        _validate_finite_number(
            self.client_read_timeout_s,
            name="client_read_timeout_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_finite_number(
            self.client_header_timeout_s,
            name="client_header_timeout_s",
            min_value=0.0,
            strictly_greater=True,
        )
        _validate_int_range(
            self.max_concurrent_connections,
            name="max_concurrent_connections",
            min_value=1,
        )
        _validate_int_range(
            self.max_sse_pre_commit_bytes,
            name="max_sse_pre_commit_bytes",
            min_value=1,
        )
        if isinstance(self.cors_allowed_origins, (str, bytes)):
            raise ValueError("cors_allowed_origins must be a list of origin strings")
        normalized_origins: List[str] = []
        for raw_origin in self.cors_allowed_origins:
            if not isinstance(raw_origin, str) or not raw_origin.strip():
                raise ValueError("cors_allowed_origins entries must be non-empty strings")
            origin = raw_origin.strip()
            normalized_origins.append("*" if origin == "*" else origin.rstrip("/"))
        self.cors_allowed_origins = normalized_origins

        if not isinstance(self.auth, Mapping):
            raise ValueError("'auth' must be a JSON object")
        auth_spec = self.auth.get("hook")
        auth_type = self.auth.get("type")
        if auth_spec is not None:
            if not isinstance(auth_spec, str) or ":" not in auth_spec.strip():
                raise ValueError(
                    f"Invalid auth.hook spec {auth_spec!r}; expected 'module:ClassName'"
                )
            spec_str = auth_spec.strip()
            if not all(part.strip() for part in spec_str.split(":", 1)):
                raise ValueError(
                    f"Invalid auth.hook spec {auth_spec!r}; expected 'module:ClassName'"
                )
            if auth_type is not None and auth_type != "custom":
                raise ValueError(
                    f"Invalid auth.type {auth_type!r} when auth.hook is configured; omit 'type' or set 'custom'"
                )
        elif auth_type is not None:
            if not isinstance(auth_type, str) or auth_type not in VALID_AUTH_TYPES:
                raise ValueError(
                    f"Invalid auth.type {auth_type!r}; expected one of {VALID_AUTH_TYPES}"
                )
        if auth_type == "bearer":
            if "tokens" in self.auth and self.auth["tokens"] is not None:
                raw_tokens = self.auth["tokens"]
                if not isinstance(raw_tokens, (list, tuple)) or not all(
                    isinstance(tok, str) and tok.strip() for tok in raw_tokens
                ):
                    raise ValueError(
                        "auth.tokens must be a list of non-empty strings"
                    )
            if "tokens_env" in self.auth and self.auth["tokens_env"] is not None:
                raw_env = self.auth["tokens_env"]
                if not isinstance(raw_env, (list, tuple)) or not all(
                    isinstance(name, str) and name.strip() for name in raw_env
                ):
                    raise ValueError(
                        "auth.tokens_env must be a list of non-empty environment variable names"
                    )
            if "exempt_paths" in self.auth and self.auth["exempt_paths"] is not None:
                raw_exempt = self.auth["exempt_paths"]
                if not isinstance(raw_exempt, (list, tuple)) or not all(
                    isinstance(path, str) and path.strip() for path in raw_exempt
                ):
                    raise ValueError(
                        "auth.exempt_paths must be a list of non-empty path strings"
                    )
        else:
            misplaced = [k for k in BEARER_AUTH_ONLY_KEYS if k in self.auth]
            if misplaced:
                raise ValueError(
                    f"Bearer auth settings {misplaced!r} require auth.type='bearer'"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RouterConfig":
        raw_backends: Sequence[Mapping[str, Any]] = raw.get("backends") or []
        backends = [BackendConfig.from_dict(item) for item in raw_backends]
        if not backends:
            raise ValueError("Config must define at least one backend in 'backends'")
        raw_origins = raw.get("cors_allowed_origins")
        if raw_origins is None:
            cors_origins: List[str] = []
        elif isinstance(raw_origins, (list, tuple)):
            cors_origins = list(raw_origins)
        else:
            raise ValueError("'cors_allowed_origins' must be a list of origin strings")
        raw_auth = raw.get("auth")
        if raw_auth is not None and not isinstance(raw_auth, Mapping):
            raise ValueError("'auth' must be a JSON object")
        return cls(
            backends=backends,
            host=str(raw.get("host", "127.0.0.1")),
            port=_validate_int_range(
                raw.get("port", 8080), name="port", min_value=0, max_value=65535
            ),
            policy=str(raw.get("policy", "local_first")),
            health_poll_interval_s=_validate_finite_number(
                raw.get("health_poll_interval_s", 10.0),
                name="health_poll_interval_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            failure_cooldown_s=_validate_finite_number(
                raw.get("failure_cooldown_s", 15.0),
                name="failure_cooldown_s",
                min_value=0.0,
            ),
            request_timeout_s=_validate_finite_number(
                raw.get("request_timeout_s", 120.0),
                name="request_timeout_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            probe_timeout_s=_validate_finite_number(
                raw.get("probe_timeout_s", 3.0),
                name="probe_timeout_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            queue_pressure_threshold=_validate_int_range(
                raw.get("queue_pressure_threshold", 16),
                name="queue_pressure_threshold",
                min_value=1,
            ),
            backend_idle_timeout_s=_validate_finite_number(
                raw.get("backend_idle_timeout_s", 300.0),
                name="backend_idle_timeout_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            max_request_body_bytes=_validate_int_range(
                raw.get("max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES),
                name="max_request_body_bytes",
                min_value=1,
            ),
            max_response_body_bytes=_validate_int_range(
                raw.get("max_response_body_bytes", DEFAULT_MAX_RESPONSE_BODY_BYTES),
                name="max_response_body_bytes",
                min_value=1,
            ),
            client_read_timeout_s=_validate_finite_number(
                raw.get("client_read_timeout_s", DEFAULT_CLIENT_READ_TIMEOUT_S),
                name="client_read_timeout_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            client_header_timeout_s=_validate_finite_number(
                raw.get("client_header_timeout_s", DEFAULT_CLIENT_HEADER_TIMEOUT_S),
                name="client_header_timeout_s",
                min_value=0.0,
                strictly_greater=True,
            ),
            max_concurrent_connections=_validate_int_range(
                raw.get("max_concurrent_connections", DEFAULT_MAX_CONCURRENT_CONNECTIONS),
                name="max_concurrent_connections",
                min_value=1,
            ),
            max_sse_pre_commit_bytes=_validate_int_range(
                raw.get("max_sse_pre_commit_bytes", DEFAULT_MAX_SSE_PRE_COMMIT_BYTES),
                name="max_sse_pre_commit_bytes",
                min_value=1,
            ),
            cors_allowed_origins=cors_origins,
            auth=dict(raw_auth or {}),
            usage_meter=dict(raw.get("usage_meter") or {}),
            scaler=dict(raw.get("scaler") or {}),
        )


def load_config(path: Path | str) -> RouterConfig:
    """Load and validate a `RouterConfig` from a JSON file."""
    cfg_path = Path(path)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {cfg_path}")
    return RouterConfig.from_dict(data)
