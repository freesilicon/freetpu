#!/usr/bin/env python3
"""Configuration schema and JSON loader for FreeTPU Router v0."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

VALID_POLICIES = ("cheapest", "local_first", "lowest_latency", "round_robin")
VALID_AUTH_TYPES = ("allow_all", "bearer")
DEFAULT_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024
DEFAULT_CLIENT_READ_TIMEOUT_S = 30.0


def sanitize_url(url: str) -> str:
    """Strip any userinfo credentials from a URL before logging or displaying."""
    parts = urlsplit(url)
    if not parts.username and not parts.password:
        return url
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return parts._replace(netloc=netloc).geturl()


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
    latency_hint_ms: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        if "local" in raw:
            is_local = bool(raw["local"])
        else:
            lowered = btype.lower()
            is_local = lowered.startswith(("pixel", "local", "mac", "edge"))

        models_raw = raw.get("models") or []
        models = [str(m).strip() for m in models_raw if str(m).strip()]

        cost_shared = raw.get("cost_per_1m_tokens_usd")
        prompt_cost = float(
            raw.get(
                "cost_per_1m_prompt_usd",
                cost_shared if cost_shared is not None else 0.0,
            )
        )
        completion_cost = float(
            raw.get(
                "cost_per_1m_completion_usd",
                cost_shared if cost_shared is not None else 0.0,
            )
        )
        if prompt_cost < 0 or completion_cost < 0:
            raise ValueError(f"Backend {backend_id!r} costs must be non-negative")

        max_hbm = float(raw.get("max_hbm_utilization", 0.95))
        if not (0.0 < max_hbm <= 1.0):
            raise ValueError(
                f"Backend {backend_id!r} max_hbm_utilization must be in (0, 1]"
            )

        max_duty = raw.get("max_duty_cycle_percent")
        max_duty_val = float(max_duty) if max_duty is not None else None

        max_inf = raw.get("max_inflight")
        max_inf_val = int(max_inf) if max_inf is not None else None
        if max_inf_val is not None and max_inf_val <= 0:
            raise ValueError(f"Backend {backend_id!r} max_inflight must be > 0")

        timeout_raw = raw.get("timeout_s")
        timeout_val = float(timeout_raw) if timeout_raw is not None else None

        lat_hint_raw = raw.get("latency_hint_ms")
        lat_hint_val = float(lat_hint_raw) if lat_hint_raw is not None else None

        return cls(
            id=backend_id,
            base_url=base_url,
            type=btype,
            local=is_local,
            spot=bool(raw.get("spot", False)),
            models=models,
            discover_models=bool(raw.get("discover_models", True)),
            cost_per_1m_prompt_usd=prompt_cost,
            cost_per_1m_completion_usd=completion_cost,
            priority=int(raw.get("priority", 100)),
            weight=max(1, int(raw.get("weight", 1))),
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
            latency_hint_ms=lat_hint_val,
            enabled=bool(raw.get("enabled", True)),
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
    client_read_timeout_s: float = DEFAULT_CLIENT_READ_TIMEOUT_S
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
            if b.id in seen_ids:
                raise ValueError(f"Duplicate backend id {b.id!r}")
            seen_ids.add(b.id)
        if not (0 <= self.port <= 65535):
            raise ValueError(f"Invalid port {self.port}")
        if self.health_poll_interval_s <= 0:
            raise ValueError("health_poll_interval_s must be > 0")
        if self.failure_cooldown_s < 0:
            raise ValueError("failure_cooldown_s must be >= 0")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")
        if self.max_request_body_bytes <= 0:
            raise ValueError("max_request_body_bytes must be > 0")
        if self.client_read_timeout_s <= 0:
            raise ValueError("client_read_timeout_s must be > 0")
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
            port=int(raw.get("port", 8080)),
            policy=str(raw.get("policy", "local_first")),
            health_poll_interval_s=float(raw.get("health_poll_interval_s", 10.0)),
            failure_cooldown_s=float(raw.get("failure_cooldown_s", 15.0)),
            request_timeout_s=float(raw.get("request_timeout_s", 120.0)),
            probe_timeout_s=float(raw.get("probe_timeout_s", 3.0)),
            queue_pressure_threshold=int(raw.get("queue_pressure_threshold", 16)),
            backend_idle_timeout_s=float(raw.get("backend_idle_timeout_s", 300.0)),
            max_request_body_bytes=int(
                raw.get("max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES)
            ),
            client_read_timeout_s=float(
                raw.get("client_read_timeout_s", DEFAULT_CLIENT_READ_TIMEOUT_S)
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
