#!/usr/bin/env python3
"""Extension hooks for FreeTPU Router v0.

Defines clean, stdlib-only interfaces (`AuthHook`, `UsageMeterHook`, `ScalerHook`)
so external wrappers and future commercial control planes can plug in auth,
structured usage metering, and Spot TPU autoscaling without forking v0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hmac
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass
class AuthContext:
    """Result returned by `AuthHook.authenticate`."""

    authorized: bool
    principal: str = "anonymous"
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    status_code: int = 401


class AuthHook(ABC):
    """Authentication and authorization hook for incoming router requests."""

    @abstractmethod
    def authenticate(
        self,
        headers: Mapping[str, str],
        path: str,
        method: str,
    ) -> AuthContext:
        """Inspect request headers/path/method and return an `AuthContext`."""


class AllowAllAuthHook(AuthHook):
    """Default open-source v0 auth hook allowing all requests."""

    def authenticate(
        self,
        headers: Mapping[str, str],
        path: str,
        method: str,
    ) -> AuthContext:
        return AuthContext(authorized=True, principal="anonymous")


class BearerTokenAuthHook(AuthHook):
    """Optional bearer-token auth hook using constant-time token comparison."""

    def __init__(
        self,
        tokens: Sequence[str] | Mapping[str, str],
        *,
        exempt_paths: Iterable[str] = ("/health",),
    ) -> None:
        if isinstance(tokens, (str, bytes)):
            raise ValueError("tokens must be a list or mapping of token strings, not a scalar string")
        if isinstance(exempt_paths, (str, bytes)):
            raise ValueError("exempt_paths must be an iterable of path strings, not a scalar string")
        if isinstance(tokens, Mapping):
            self._token_principals: Dict[str, str] = {
                tok.strip(): str(principal)
                for tok, principal in tokens.items()
                if isinstance(tok, str) and tok.strip()
            }
        else:
            self._token_principals = {
                tok.strip(): f"token-{idx + 1}"
                for idx, tok in enumerate(tokens)
                if isinstance(tok, str) and tok.strip()
            }
        if not self._token_principals:
            raise ValueError(
                "BearerTokenAuthHook requires at least one non-empty bearer token"
            )
        self._exempt_paths = set(exempt_paths)

    def authenticate(
        self,
        headers: Mapping[str, str],
        path: str,
        method: str,
    ) -> AuthContext:
        auth_header = ""
        for k, v in headers.items():
            if k.lower() == "authorization":
                auth_header = v.strip()
                break

        if self._token_principals and auth_header.lower().startswith("bearer "):
            presented = auth_header[7:].strip()
            matched_principal: Optional[str] = None
            for token, principal in self._token_principals.items():
                if hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8")):
                    matched_principal = principal
            if matched_principal is not None:
                return AuthContext(
                    authorized=True,
                    principal=matched_principal,
                    metadata={"authenticated": True},
                )

        if path in self._exempt_paths:
            return AuthContext(
                authorized=True,
                principal="health-probe",
                metadata={"authenticated": False, "exempt_path": True},
            )

        if not self._token_principals:
            return AuthContext(
                authorized=False,
                error_message="No bearer tokens configured on router",
                status_code=401,
            )

        if not auth_header.lower().startswith("bearer "):
            return AuthContext(
                authorized=False,
                error_message="Missing or malformed Authorization: Bearer token",
                status_code=401,
            )

        return AuthContext(
            authorized=False,
            error_message="Invalid bearer token",
            status_code=401,
        )


@dataclass
class UsageEvent:
    """Structured per-request usage and cost record emitted to `UsageMeterHook`."""

    request_id: str
    model: str
    backend_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    timestamp_unix: float = field(default_factory=time.time)
    ttft_ms: Optional[float] = None
    stream: bool = False
    failovers: int = 0
    principal: str = "anonymous"
    status_code: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UsageMeterHook(ABC):
    """Usage accounting hook invoked after each completed inference request."""

    @abstractmethod
    def record_usage(self, event: UsageEvent) -> None:
        """Persist or forward a completed `UsageEvent`."""


class NoOpUsageMeterHook(UsageMeterHook):
    """Default no-op metering hook."""

    def record_usage(self, event: UsageEvent) -> None:
        return None


class CallbackUsageMeterHook(UsageMeterHook):
    """Usage metering hook invoking an in-process Python callback."""

    def __init__(self, callback: Callable[[UsageEvent], None]) -> None:
        self._callback = callback

    def record_usage(self, event: UsageEvent) -> None:
        self._callback(event)


class JsonlUsageMeterHook(UsageMeterHook):
    """Thread-safe JSONL usage meter with optional secondary Python callback."""

    def __init__(
        self,
        path: Path | str,
        *,
        callback: Optional[Callable[[UsageEvent], None]] = None,
        error_log_interval_s: float = 30.0,
        error_stream: Optional[Any] = None,
    ) -> None:
        self.path = Path(path)
        self._callback = callback
        self._lock = threading.Lock()
        self.events_written: int = 0
        self.write_errors: int = 0
        self._error_log_interval_s = max(0.0, float(error_log_interval_s))
        self._error_stream = error_stream
        self._last_error_log_monotonic: Optional[float] = None
        self._suppressed_errors: int = 0

    def record_usage(self, event: UsageEvent) -> None:
        line = json.dumps(event.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                self.events_written += 1
            except OSError as exc:
                self.write_errors += 1
                now = time.monotonic()
                if (
                    self._last_error_log_monotonic is None
                    or (now - self._last_error_log_monotonic) >= self._error_log_interval_s
                ):
                    suppressed = self._suppressed_errors
                    self._suppressed_errors = 0
                    self._last_error_log_monotonic = now
                    stream = (
                        self._error_stream
                        if self._error_stream is not None
                        else sys.stderr
                    )
                    suffix = (
                        f" ({suppressed} suppressed since last warning)"
                        if suppressed > 0
                        else ""
                    )
                    try:
                        print(
                            f"[freetpu-router] WARNING: failed to append JSONL usage record to "
                            f"{self.path}: {type(exc).__name__}: {exc} "
                            f"(total_write_errors={self.write_errors}){suffix}",
                            file=stream,
                        )
                    except OSError:
                        pass
                else:
                    self._suppressed_errors += 1
                raise
        if self._callback is not None:
            self._callback(event)


@dataclass
class QueuePressureEvent:
    """Lifecycle signal emitted when queue/in-flight pressure or exhaustion occurs."""

    model: str
    active_inflight: int
    healthy_backends: int
    total_backends: int
    reason: str
    timestamp_unix: float = field(default_factory=time.time)


@dataclass
class BackendIdleEvent:
    """Lifecycle signal emitted when a backend has been idle past threshold."""

    backend_id: str
    backend_type: str
    spot: bool
    idle_seconds: float
    timestamp_unix: float = field(default_factory=time.time)


class ScalerHook(ABC):
    """Lifecycle interface for external Spot TPU and worker autoscalers."""

    @abstractmethod
    def on_queue_pressure(self, event: QueuePressureEvent) -> None:
        """Called when active in-flight demand exceeds threshold or pool is exhausted."""

    @abstractmethod
    def on_backend_idle(self, event: BackendIdleEvent) -> None:
        """Called when a backend remains idle beyond `idle_timeout_s`."""


class NoOpScalerHook(ScalerHook):
    """Default no-op autoscaler hook."""

    def on_queue_pressure(self, event: QueuePressureEvent) -> None:
        return None

    def on_backend_idle(self, event: BackendIdleEvent) -> None:
        return None


_HOOK_ROLE_REQUIRED_METHODS: Dict[str, Tuple[str, ...]] = {
    "auth": ("authenticate",),
    "usage_meter": ("record_usage",),
    "scaler": ("on_queue_pressure", "on_backend_idle"),
}


def validate_hook_for_role(hook: Any, role: str) -> Any:
    """Verify that `hook` exposes every callable method required for `role`."""
    required = _HOOK_ROLE_REQUIRED_METHODS.get(role)
    if required is None:
        raise ValueError(f"Unknown hook role {role!r}")
    missing = [
        name for name in required if not callable(getattr(hook, name, None))
    ]
    if missing:
        raise ValueError(
            f"Configured {role} hook {type(hook).__name__!r} is missing required "
            f"callable method(s): {', '.join(missing)}"
        )
    return hook


def load_hook_from_spec(
    spec: str,
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    role: Optional[str] = None,
) -> Any:
    """Instantiate a hook class from a `'module.path:ClassName'` spec."""
    if ":" not in spec:
        raise ValueError(f"Hook spec must be 'module:ClassName', got {spec!r}")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name.strip())
    cls = getattr(module, attr_name.strip())
    if not callable(cls):
        raise ValueError(f"Hook target {spec!r} is not callable")
    instance = cls(**dict(kwargs or {}))
    if role is not None:
        validate_hook_for_role(instance, role)
    return instance
