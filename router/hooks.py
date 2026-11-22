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
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


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
        if isinstance(tokens, Mapping):
            self._token_principals: Dict[str, str] = {
                str(tok): str(principal) for tok, principal in tokens.items() if tok
            }
        else:
            self._token_principals = {
                str(tok): f"token-{idx + 1}"
                for idx, tok in enumerate(tokens)
                if tok
            }
        self._exempt_paths = set(exempt_paths)

    def authenticate(
        self,
        headers: Mapping[str, str],
        path: str,
        method: str,
    ) -> AuthContext:
        if path in self._exempt_paths:
            return AuthContext(authorized=True, principal="health-probe")

        if not self._token_principals:
            return AuthContext(
                authorized=False,
                error_message="No bearer tokens configured on router",
                status_code=401,
            )

        auth_header = ""
        for k, v in headers.items():
            if k.lower() == "authorization":
                auth_header = v.strip()
                break

        if not auth_header.lower().startswith("bearer "):
            return AuthContext(
                authorized=False,
                error_message="Missing or malformed Authorization: Bearer token",
                status_code=401,
            )

        presented = auth_header[7:].strip()
        matched_principal: Optional[str] = None
        for token, principal in self._token_principals.items():
            if hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8")):
                matched_principal = principal

        if matched_principal is None:
            return AuthContext(
                authorized=False,
                error_message="Invalid bearer token",
                status_code=401,
            )

        return AuthContext(authorized=True, principal=matched_principal)


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
    ) -> None:
        self.path = Path(path)
        self._callback = callback
        self._lock = threading.Lock()
        self.events_written: int = 0

    def record_usage(self, event: UsageEvent) -> None:
        line = json.dumps(event.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            self.events_written += 1
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


def load_hook_from_spec(
    spec: str,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Instantiate a hook class from a `'module.path:ClassName'` spec."""
    if ":" not in spec:
        raise ValueError(f"Hook spec must be 'module:ClassName', got {spec!r}")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name.strip())
    cls = getattr(module, attr_name.strip())
    return cls(**dict(kwargs or {}))
