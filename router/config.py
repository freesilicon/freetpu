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
DEFAULT_MAX_TOTAL_REQUEST_MEMORY_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_RESPONSE_MEMORY_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_JSON_DEPTH = 64
DEFAULT_MAX_JSON_NODES = 32768
DEFAULT_MAX_RESPONSE_JSON_NODES = 524288
DEFAULT_CLIENT_READ_TIMEOUT_S = 30.0
DEFAULT_CLIENT_HEADER_TIMEOUT_S = 10.0
DEFAULT_MAX_CONCURRENT_CONNECTIONS = 256
DEFAULT_MAX_SSE_PRE_COMMIT_BYTES = 1024 * 1024


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in value)


def _validate_credential_string(
    value: Any,
    *,
    context: str,
    field_name: str,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{context} field {field_name!r} must be a string")
    if _has_control_chars(value):
        raise ValueError(
            f"{context} field {field_name!r} must not contain control characters or newlines"
        )


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
    """Strip userinfo credentials, query strings, and fragments before logging or displaying."""
    try:
        parts = urlsplit(url)
    except ValueError:
        cleaned = url.split("#", 1)[0].split("?", 1)[0]
        if "://" in cleaned:
            scheme, rest = cleaned.split("://", 1)
            if "/" in rest:
                authority, tail = rest.split("/", 1)
                if "@" in authority:
                    authority = authority.rsplit("@", 1)[1]
                return f"{scheme}://{authority}/{tail}"
            if "@" in rest:
                rest = rest.rsplit("@", 1)[1]
            return f"{scheme}://{rest}"
        return cleaned
    netloc = parts.netloc.rsplit("@", 1)[1] if "@" in parts.netloc else parts.netloc
    if (
        netloc == parts.netloc
        and not parts.query
        and not parts.fragment
        and "?" not in url
        and "#" not in url
    ):
        return url
    return parts._replace(netloc=netloc, query="", fragment="").geturl()


def normalize_origin(origin: str) -> Optional[tuple[str, str, int]]:
    """Parse and normalize an HTTP(S) origin to `(scheme, hostname, port)`, or None if invalid."""
    cleaned = origin.strip()
    if not cleaned or any(ch.isspace() for ch in cleaned):
        return None
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in VALID_URL_SCHEMES:
        return None
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is not None and not (1 <= port <= 65535):
        return None
    resolved_port = port if port is not None else (443 if scheme == "https" else 80)
    return scheme, host, resolved_port


def validate_http_url(
    url: str,
    *,
    field_name: str = "url",
    allow_query: bool = True,
) -> str:
    """Validate that a backend or exporter URL uses an explicit http/https scheme, host, and port."""
    cleaned = url.strip()
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: malformed URL"
        ) from None
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
    if parts.fragment or "#" in cleaned:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: fragments are not supported"
        )
    if not allow_query and (parts.query or "?" in cleaned):
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: query parameters are not supported"
        )
    try:
        port = parts.port
    except ValueError:
        raise ValueError(
            f"Invalid {field_name} {sanitize_url(cleaned)!r}: invalid port"
        ) from None
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
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Backend 'id' must be a non-empty string")
        if self.id != self.id.strip() or any(
            ord(ch) < 0x21 or ord(ch) > 0x7E for ch in self.id
        ):
            raise ValueError(
                f"Backend 'id' {self.id!r} must contain only printable ASCII characters "
                "without whitespace or control characters"
            )
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
            self.base_url,
            field_name=f"backend {self.id!r} base_url",
            allow_query=False,
        ).rstrip("/")
        if self.tpu_exporter_url:
            self.tpu_exporter_url = validate_http_url(
                self.tpu_exporter_url,
                field_name=f"backend {self.id!r} tpu_exporter_url",
                allow_query=True,
            )
        ctx = f"Backend {self.id!r}"
        if not isinstance(self.models, list) or not all(
            isinstance(m, str) for m in self.models
        ):
            raise ValueError(
                f"{ctx} field 'models' must be a list of strings, got {self.models!r}"
            )
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
        if self.api_key is not None:
            _validate_credential_string(self.api_key, context=ctx, field_name="api_key")
        if self.api_key_env is not None:
            _validate_credential_string(
                self.api_key_env, context=ctx, field_name="api_key_env"
            )
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
        ctx = f"Backend {self.id!r}"
        if self.api_key:
            _validate_credential_string(self.api_key, context=ctx, field_name="api_key")
            return self.api_key
        if self.api_key_env:
            raw_val = os.environ.get(self.api_key_env, "")
            if not raw_val:
                return None
            _validate_credential_string(
                raw_val, context=ctx, field_name=f"api_key_env({self.api_key_env})"
            )
            val = raw_val.strip()
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

        if "models" not in raw or raw["models"] is None:
            models: List[str] = []
        else:
            models_raw = raw["models"]
            if not isinstance(models_raw, list) or not all(
                isinstance(m, str) for m in models_raw
            ):
                raise ValueError(
                    f"{ctx} field 'models' must be a JSON array of strings, got {models_raw!r}"
                )
            models = [m.strip() for m in models_raw if m.strip()]

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

        api_key_raw = raw.get("api_key")
        api_key_val: Optional[str] = None
        if api_key_raw is not None:
            _validate_credential_string(api_key_raw, context=ctx, field_name="api_key")
            api_key_val = api_key_raw or None

        api_key_env_raw = raw.get("api_key_env")
        api_key_env_val: Optional[str] = None
        if api_key_env_raw is not None:
            _validate_credential_string(
                api_key_env_raw, context=ctx, field_name="api_key_env"
            )
            api_key_env_val = api_key_env_raw.strip() or None

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
            api_key=api_key_val,
            api_key_env=api_key_env_val,
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
    max_total_request_memory_bytes: int = DEFAULT_MAX_TOTAL_REQUEST_MEMORY_BYTES
    max_total_response_memory_bytes: int = DEFAULT_MAX_TOTAL_RESPONSE_MEMORY_BYTES
    max_response_body_bytes: int = DEFAULT_MAX_RESPONSE_BODY_BYTES
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH
    max_json_nodes: int = DEFAULT_MAX_JSON_NODES
    max_response_json_nodes: int = DEFAULT_MAX_RESPONSE_JSON_NODES
    client_read_timeout_s: float = DEFAULT_CLIENT_READ_TIMEOUT_S
    client_header_timeout_s: float = DEFAULT_CLIENT_HEADER_TIMEOUT_S
    max_concurrent_connections: int = DEFAULT_MAX_CONCURRENT_CONNECTIONS
    max_sse_pre_commit_bytes: int = DEFAULT_MAX_SSE_PRE_COMMIT_BYTES
    cors_allowed_origins: List[str] = field(default_factory=list)
    public_origin: Optional[str] = None
    trust_proxy_headers: bool = False
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
            self.max_total_request_memory_bytes,
            name="max_total_request_memory_bytes",
            min_value=1,
        )
        _validate_int_range(
            self.max_total_response_memory_bytes,
            name="max_total_response_memory_bytes",
            min_value=1,
        )
        _validate_int_range(
            self.max_response_body_bytes,
            name="max_response_body_bytes",
            min_value=1,
        )
        _validate_int_range(
            self.max_header_bytes,
            name="max_header_bytes",
            min_value=1,
        )
        _validate_int_range(
            self.max_json_depth,
            name="max_json_depth",
            min_value=1,
        )
        _validate_int_range(
            self.max_json_nodes,
            name="max_json_nodes",
            min_value=1,
        )
        _validate_int_range(
            self.max_response_json_nodes,
            name="max_response_json_nodes",
            min_value=1,
        )
        if not isinstance(self.trust_proxy_headers, bool):
            raise ValueError(
                f"trust_proxy_headers must be a boolean, got {self.trust_proxy_headers!r}"
            )
        if self.public_origin is not None:
            if not isinstance(self.public_origin, str) or normalize_origin(self.public_origin) is None:
                raise ValueError(
                    f"Invalid public_origin {self.public_origin!r}; expected 'http(s)://host[:port]'"
                )
            self.public_origin = self.public_origin.strip().rstrip("/")
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
            if origin == "*":
                normalized_origins.append("*")
                continue
            parsed_origin = normalize_origin(origin)
            if parsed_origin is not None:
                scheme, host, port = parsed_origin
                host_fmt = f"[{host}]" if ":" in host else host
                default_port = 443 if scheme == "https" else 80
                canonical = (
                    f"{scheme}://{host_fmt}"
                    if port == default_port
                    else f"{scheme}://{host_fmt}:{port}"
                )
                normalized_origins.append(canonical)
            else:
                normalized_origins.append(origin.rstrip("/"))
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
            max_total_request_memory_bytes=_validate_int_range(
                raw.get(
                    "max_total_request_memory_bytes",
                    DEFAULT_MAX_TOTAL_REQUEST_MEMORY_BYTES,
                ),
                name="max_total_request_memory_bytes",
                min_value=1,
            ),
            max_total_response_memory_bytes=_validate_int_range(
                raw.get(
                    "max_total_response_memory_bytes",
                    DEFAULT_MAX_TOTAL_RESPONSE_MEMORY_BYTES,
                ),
                name="max_total_response_memory_bytes",
                min_value=1,
            ),
            max_response_body_bytes=_validate_int_range(
                raw.get("max_response_body_bytes", DEFAULT_MAX_RESPONSE_BODY_BYTES),
                name="max_response_body_bytes",
                min_value=1,
            ),
            max_header_bytes=_validate_int_range(
                raw.get("max_header_bytes", DEFAULT_MAX_HEADER_BYTES),
                name="max_header_bytes",
                min_value=1,
            ),
            max_json_depth=_validate_int_range(
                raw.get("max_json_depth", DEFAULT_MAX_JSON_DEPTH),
                name="max_json_depth",
                min_value=1,
            ),
            max_json_nodes=_validate_int_range(
                raw.get("max_json_nodes", DEFAULT_MAX_JSON_NODES),
                name="max_json_nodes",
                min_value=1,
            ),
            max_response_json_nodes=_validate_int_range(
                raw.get("max_response_json_nodes", DEFAULT_MAX_RESPONSE_JSON_NODES),
                name="max_response_json_nodes",
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
            public_origin=(
                str(raw["public_origin"]).strip()
                if raw.get("public_origin") is not None
                else None
            ),
            trust_proxy_headers=_require_bool(
                raw, "trust_proxy_headers", False, context="RouterConfig"
            ),
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
