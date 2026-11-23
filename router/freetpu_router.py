#!/usr/bin/env python3
"""FreeTPU Router v0 — OpenAI-compatible router for heterogeneous TPU silicon.

Routes `/v1/chat/completions`, `/v1/completions`, and `/v1/models` across wired
Pixel Tensor G5 llama-server clusters, local CPU/GPU/Mac hosts, and Cloud TPU
v5e/v6e workers with zero third-party dependencies (Python standard library only).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import errno
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import re
import selectors
import socket
import ssl
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
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
    normalize_origin,
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
    validate_hook_for_role,
)
from router.metrics import RouterMetrics  # noqa: E402
from router.tpu_telemetry import (  # noqa: E402
    ClientDisconnectedError,
    DeadlineSocket,
    TpuExporterTelemetry,
    evaluate_tpu_readiness,
    scrape_tpu_exporter,
    wait_sockets_readable,
)

__version__ = "0.1.0"

RETRYABLE_STATUS_CODES = (502, 503, 504)
UNKNOWN_MODEL_LABEL = "_unknown"
OTHER_MODEL_LABEL = "_other"
MAX_METRIC_MODEL_LABELS = 256
MAX_RR_CURSORS = 1024
MAX_PROBE_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_RESPONSE_BYTES = 1024 * 1024
MAX_MODEL_ID_LEN = 256
REQUEST_MEMORY_PARSE_MULTIPLIER = 4
_UTF8_4BYTE_RE = re.compile(rb"[\xf0-\xff]")
_API_VERSION_SUFFIX_RE = re.compile(r"/v\d+(?:alpha|beta\d*)?$")
_LOG_CONTROL_CHAR_TABLE = str.maketrans(
    {c: f"\\x{c:02x}" for c in (*range(0x00, 0x20), *range(0x7F, 0xA0))}
)


def _safe_probe_error(exc: BaseException) -> str:
    """Map probe/upstream exceptions to fixed safe diagnostics without echoing secrets."""
    if isinstance(exc, TimeoutError):
        return "probe_timeout"
    if isinstance(exc, _UpstreamResponseTooLargeError):
        return "probe_response_too_large"
    if isinstance(exc, http.client.InvalidURL):
        return "invalid_probe_url"
    if isinstance(exc, http.client.IncompleteRead):
        return "probe_incomplete_response"
    if isinstance(exc, (http.client.BadStatusLine, http.client.RemoteDisconnected)):
        return "probe_connection_closed"
    if isinstance(exc, http.client.HTTPException):
        return f"probe_http_error:{type(exc).__name__}"
    if isinstance(exc, ssl.SSLError):
        return "probe_tls_error"
    if isinstance(exc, ConnectionRefusedError):
        return "probe_connection_refused"
    if isinstance(exc, ConnectionResetError):
        return "probe_connection_reset"
    if isinstance(exc, OSError):
        return f"probe_network_error:{type(exc).__name__}"
    if isinstance(exc, ValueError):
        return "invalid_request_header"
    return f"probe_exception:{type(exc).__name__}"


def _extract_models_list(parsed: Any) -> Optional[List[str]]:
    """Validate an OpenAI `/v1/models` JSON response and return discovered model IDs."""
    items: Optional[List[Any]] = None
    if isinstance(parsed, Mapping):
        raw_data = parsed.get("data")
        if isinstance(raw_data, list):
            items = raw_data
    elif isinstance(parsed, list):
        items = parsed
    if items is None:
        return None
    discovered: List[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        raw_id = item.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            discovered.append(raw_id.strip())
    return discovered


class _UpstreamResponseTooLargeError(http.client.HTTPException):
    """Raised when an upstream non-streaming HTTP body exceeds the byte cap."""


class _MemoryBudgetExceededError(Exception):
    """Raised when buffering a response would exceed the shared router memory budget."""


class _HeaderSizeLimitExceeded(Exception):
    """Raised when cumulative inbound request header bytes exceed the configured cap."""


class _JsonStructureLimitError(ValueError):
    """Raised when inbound JSON exceeds nesting depth or structural node budgets."""


class _MalformedSSEPayloadError(Exception):
    """Raised when an upstream SSE data frame contains invalid or non-object JSON before commit."""


def _validate_json_structure(
    text: str | bytes | bytearray,
    *,
    max_depth: int,
    max_nodes: int,
    containers_out: Optional[List[int]] = None,
) -> int:
    """Validate JSON nesting depth and structural count in O(1) working memory.

    Accepts UTF-8 `bytes`/`bytearray` (avoiding decoded string allocation before
    validation) or `str`, and returns the number of structural JSON tokens seen.
    When `containers_out` is provided (a 1-element list), stores the count of
    object/array container openings (`{` and `[`) seen outside strings.
    """
    n = len(text)
    i = 0
    depth = 0
    nodes = 0
    containers = 0
    if isinstance(text, (bytes, bytearray)):
        # UTF-8 is ASCII-compatible: structural ASCII bytes never appear inside
        # multi-byte UTF-8 sequences (all continuation/lead bytes are >= 0x80).
        while i < n:
            ch = text[i]
            if ch == 34:  # '"'
                i += 1
                while True:
                    q = text.find(b'"', i)
                    if q == -1:
                        if containers_out is not None:
                            containers_out[0] = containers
                        return nodes
                    bs = 0
                    j = q - 1
                    while j >= i and text[j] == 92:  # '\\'
                        bs += 1
                        j -= 1
                    if (bs & 1) == 0:
                        i = q + 1
                        break
                    i = q + 1
                continue
            if ch == 123 or ch == 91:  # '{' or '['
                depth += 1
                if depth > max_depth:
                    raise _JsonStructureLimitError(
                        f"JSON nesting depth exceeds maximum limit of {max_depth}"
                    )
                containers += 1
                nodes += 1
                if nodes > max_nodes:
                    raise _JsonStructureLimitError(
                        f"JSON structure exceeds maximum limit of {max_nodes} elements"
                    )
            elif ch == 125 or ch == 93:  # '}' or ']'
                if depth > 0:
                    depth -= 1
            elif ch == 44 or ch == 58:  # ',' or ':'
                nodes += 1
                if nodes > max_nodes:
                    raise _JsonStructureLimitError(
                        f"JSON structure exceeds maximum limit of {max_nodes} elements"
                    )
            i += 1
        if containers_out is not None:
            containers_out[0] = containers
        return nodes

    while i < n:
        ch_s = text[i]
        if ch_s == '"':
            # Skip quoted strings via C-level str.find with zero regex backtracking state.
            i += 1
            while True:
                q = text.find('"', i)
                if q == -1:
                    # Unclosed string literal; let json.loads report exact JSONDecodeError.
                    if containers_out is not None:
                        containers_out[0] = containers
                    return nodes
                bs = 0
                j = q - 1
                while j >= i and text[j] == "\\":
                    bs += 1
                    j -= 1
                if (bs & 1) == 0:
                    i = q + 1
                    break
                i = q + 1
            continue
        if ch_s == "{" or ch_s == "[":
            depth += 1
            if depth > max_depth:
                raise _JsonStructureLimitError(
                    f"JSON nesting depth exceeds maximum limit of {max_depth}"
                )
            containers += 1
            nodes += 1
            if nodes > max_nodes:
                raise _JsonStructureLimitError(
                    f"JSON structure exceeds maximum limit of {max_nodes} elements"
                )
        elif ch_s == "}" or ch_s == "]":
            if depth > 0:
                depth -= 1
        elif ch_s == "," or ch_s == ":":
            nodes += 1
            if nodes > max_nodes:
                raise _JsonStructureLimitError(
                    f"JSON structure exceeds maximum limit of {max_nodes} elements"
                )
        i += 1
    if containers_out is not None:
        containers_out[0] = containers
    return nodes


def _estimate_json_parse_memory(
    raw: bytes,
    *,
    nodes: int = 0,
    containers: int = 0,
) -> int:
    """Estimate peak CPython working-set bytes while decoding and parsing UTF-8 JSON.

    Accounts for PEP-393 flexible string widening (UCS-1 / UCS-2 / UCS-4),
    JSON `\\uXXXX` surrogate escapes, and dense parsed container/value overhead
    while `raw`, decoded text, and parsed objects coexist in memory.
    """
    n = len(raw)
    if n <= 0:
        return 1
    has_u_escape = (b"\\u" in raw) or (b"\\U" in raw)
    if raw.isascii():
        base_mult = 9 if has_u_escape else REQUEST_MEMORY_PARSE_MULTIPLIER
    elif _UTF8_4BYTE_RE.search(raw) is not None:
        base_mult = 10
    elif has_u_escape:
        base_mult = 10
    else:
        base_mult = 6
    if nodes > 64:
        if containers > 0:
            separators = max(0, nodes - containers)
            dense_node_overhead = (containers * 240) + (separators * 64)
        else:
            dense_node_overhead = nodes * 128
    else:
        dense_node_overhead = 0
    return max(n * base_mult, n * (base_mult - 1) + dense_node_overhead)


def _read_bounded_http_response(
    resp: http.client.HTTPResponse,
    upstream_sock: Any,
    *,
    max_bytes: int,
    deadline: float,
    label: str = "Upstream response",
    chunk_size: int = 65536,
    reserve_fn: Optional[Callable[[int], bool]] = None,
    release_fn: Optional[Callable[[int], None]] = None,
    reserved_out: Optional[List[int]] = None,
) -> bytes:
    """Incrementally read an HTTP response body under byte cap, memory budget, and elapsed deadline."""
    declared_len: Optional[int] = None
    reserved_local = 0
    if not resp.chunked:
        raw_len = resp.getheader("Content-Length")
        if raw_len is not None:
            try:
                parsed_len = int(raw_len.strip())
            except ValueError:
                parsed_len = -1
            if parsed_len >= 0:
                declared_len = parsed_len
                if declared_len > max_bytes:
                    raise _UpstreamResponseTooLargeError(
                        f"{label} Content-Length ({declared_len} bytes) exceeded {max_bytes} byte limit"
                    )
                if declared_len > 0 and reserve_fn is not None:
                    if not reserve_fn(declared_len):
                        raise _MemoryBudgetExceededError(
                            "Router response memory budget exhausted"
                        )
                    reserved_local += declared_len
                    if reserved_out is not None:
                        reserved_out[0] += declared_len
    buf = bytearray()
    total = 0
    while True:
        rem = deadline - time.monotonic()
        if rem <= 0:
            buf.clear()
            raise TimeoutError(f"{label} exceeded elapsed deadline")
        if isinstance(upstream_sock, DeadlineSocket):
            upstream_sock.deadline = deadline
        elif upstream_sock is not None:
            try:
                upstream_sock.settimeout(rem)
            except OSError:
                pass
        budget_left = (max_bytes - total) + 1
        if budget_left <= 0:
            buf.clear()
            raise _UpstreamResponseTooLargeError(
                f"{label} exceeded {max_bytes} byte limit"
            )
        chunk = resp.read1(min(chunk_size, budget_left))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            buf.clear()
            raise _UpstreamResponseTooLargeError(
                f"{label} exceeded {max_bytes} byte limit"
            )
        if reserve_fn is not None and total > reserved_local:
            delta = total - reserved_local
            if not reserve_fn(delta):
                buf.clear()
                raise _MemoryBudgetExceededError(
                    "Router response memory budget exhausted"
                )
            reserved_local += delta
            if reserved_out is not None:
                reserved_out[0] += delta
        buf.extend(chunk)
    if (
        not resp.chunked
        and resp.status not in (204, 304)
        and not (100 <= resp.status < 200)
    ):
        if resp.length is not None and resp.length > 0:
            buf.clear()
            raise http.client.IncompleteRead(b"", resp.length)
        if declared_len is not None and total < declared_len:
            buf.clear()
            raise http.client.IncompleteRead(b"", declared_len - total)
    copy_extra = 0
    if total > 0 and reserve_fn is not None:
        copy_extra = max(0, (2 * total) - reserved_local)
        if copy_extra > 0:
            if not reserve_fn(copy_extra):
                buf.clear()
                raise _MemoryBudgetExceededError(
                    "Router response memory budget exhausted"
                )
            reserved_local += copy_extra
            if reserved_out is not None:
                reserved_out[0] += copy_extra
    try:
        raw = bytes(buf)
    finally:
        buf.clear()
        if copy_extra > 0 and release_fn is not None:
            release_fn(copy_extra)
            if reserved_out is not None:
                reserved_out[0] = max(0, reserved_out[0] - copy_extra)
    return raw
_CORS_ALLOWED_HEADERS = ", ".join(
    (
        "Authorization",
        "Content-Type",
        "X-FreeTPU-Policy",
        "X-Request-ID",
        "OpenAI-Organization",
        "OpenAI-Project",
        "OpenAI-Beta",
        "X-Stainless-OS",
        "X-Stainless-Lang",
        "X-Stainless-Package-Version",
        "X-Stainless-Runtime",
        "X-Stainless-Runtime-Version",
        "X-Stainless-Arch",
        "X-Stainless-Retry-Count",
        "X-Stainless-Timeout",
        "X-Stainless-Read-Timeout",
        "X-Stainless-Helper-Method",
    )
)


def _join_backend_url(
    base_url: str,
    endpoint: str,
    *,
    strip_api_version: bool = False,
) -> Tuple[str, str, int, str]:
    validated = validate_http_url(
        base_url, field_name="backend base_url", allow_query=False
    )
    parsed = urlsplit(validated)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    ep = endpoint if endpoint.startswith("/") else "/" + endpoint
    if strip_api_version:
        base_root = _API_VERSION_SUFFIX_RE.sub("", base_path)
        full_path = base_root + ep
    elif ep.startswith("/v1/") and base_path.endswith("/v1"):
        full_path = base_path + ep[3:]
    elif ep == "/v1" and base_path.endswith("/v1"):
        full_path = base_path
    else:
        full_path = base_path + ep
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
        and backend.allow_insecure_http is not True
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
        reserve_fn: Optional[Callable[[int], bool]] = None,
        available_fn: Optional[Callable[[], int]] = None,
        reserved_out: Optional[List[int]] = None,
    ) -> None:
        self._resp = resp
        self._sock = upstream_sock
        self._chunk_size = chunk_size
        self._reserve_fn = reserve_fn
        self._available_fn = available_fn
        self._reserved_out = reserved_out
        self._buf = bytearray()
        self._eof = False

    def drain_buffer(self) -> Tuple[bytes, bool]:
        """Return any buffered unread bytes and EOF flag, clearing internal buffer."""
        out = bytes(self._buf)
        self._buf.clear()
        return out, self._eof

    def readline(
        self,
        *,
        deadline: Optional[float] = None,
        max_bytes: Optional[int] = None,
    ) -> Tuple[bytes, bool]:
        """Read one WHATWG SSE line (`...\\r\\n`, `...\\n`, or `...\\r`) or EOF (`b""`).

        Returns `(line, exceeded_max_bytes)`. Raises `TimeoutError` if `deadline`
        expires before a line terminator or EOF arrives.
        """
        while True:
            nl_idx = self._buf.find(b"\n")
            cr_idx = self._buf.find(b"\r")
            end: Optional[int] = None
            if cr_idx != -1 and (nl_idx == -1 or cr_idx < nl_idx):
                if cr_idx + 1 < len(self._buf) and self._buf[cr_idx + 1] == 10:
                    end = cr_idx + 2
                else:
                    end = cr_idx + 1
            elif nl_idx != -1:
                end = nl_idx + 1

            if end is not None:
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
                    raise TimeoutError("Timed out waiting for SSE data")
                if isinstance(self._sock, DeadlineSocket):
                    self._sock.deadline = deadline
                elif self._sock is not None:
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
            if self._available_fn is not None:
                avail = self._available_fn()
                to_read = min(to_read, max(1, avail))

            chunk = self._resp.read1(to_read)
            if not chunk:
                if (
                    not self._resp.chunked
                    and self._resp.status not in (204, 304)
                    and not (100 <= self._resp.status < 200)
                    and self._resp.length is not None
                    and self._resp.length > 0
                ):
                    raise http.client.IncompleteRead(bytes(self._buf), self._resp.length)
                self._eof = True
                if self._buf:
                    if max_bytes is not None and len(self._buf) > max_bytes:
                        return b"", True
                    out = bytes(self._buf)
                    self._buf.clear()
                    return out, False
                return b"", False

            if self._reserve_fn is not None:
                if not self._reserve_fn(len(chunk)):
                    raise _MemoryBudgetExceededError(
                        "Router response memory budget exhausted"
                    )
                if self._reserved_out is not None:
                    self._reserved_out[0] += len(chunk)
            self._buf.extend(chunk)


def _parse_inference_request_fields(
    raw_req: bytes,
    path: str,
) -> Tuple[
    Optional[Tuple[str, bool, int, int, bool]],
    Optional[Tuple[int, Optional[str], str]],
]:
    """Decode and validate an inbound inference JSON body in an isolated scope.

    Returns `((model, stream, est_prompt_tokens, expected_choices, expect_final_usage), None)`
    on success or `(None, (status_code, error_code, message))` on validation failure.
    Running inside a dedicated function guarantees that the decoded `str`, JSON
    `dict`, and any nested user fields (such as `stream_options`) go out of scope
    immediately before the router releases its JSON parse memory reservation.
    """
    try:
        req_text = raw_req.decode("utf-8")
        req_payload = json.loads(req_text)
        del req_text
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, (400, None, f"Malformed JSON request: {exc}")

    if not isinstance(req_payload, dict):
        return None, (400, None, "Request JSON body must be an object")

    raw_model = req_payload.get("model")
    if not isinstance(raw_model, str):
        return None, (400, "invalid_model_id", "Field 'model' must be a non-empty string")
    model = raw_model.strip()
    if not model:
        return None, (400, "invalid_model_id", "Field 'model' is required")
    if len(model) > MAX_MODEL_ID_LEN or any(
        ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in model
    ):
        return None, (
            400,
            "invalid_model_id",
            f"Field 'model' must be at most {MAX_MODEL_ID_LEN} printable characters",
        )

    raw_stream = req_payload.get("stream", False)
    if raw_stream is None:
        stream = False
    elif isinstance(raw_stream, bool):
        stream = raw_stream
    else:
        return None, (
            400,
            "invalid_stream_flag",
            "Field 'stream' must be a JSON boolean (true or false)",
        )

    est_prompt_tokens = _estimate_prompt_tokens(req_payload)
    expected_choices = _expected_stream_choices(path, req_payload)
    raw_stream_opts = req_payload.get("stream_options")
    expect_final_usage = bool(
        stream
        and isinstance(raw_stream_opts, Mapping)
        and raw_stream_opts.get("include_usage") is True
    )
    return (model, stream, est_prompt_tokens, expected_choices, expect_final_usage), None


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
        if not (isinstance(choice, dict) or isinstance(choice, Mapping)):
            continue
        text_val = choice.get("text")
        if isinstance(text_val, str) and text_val:
            parts.append(text_val)
        for container_key in ("delta", "message"):
            container = choice.get(container_key)
            if not (isinstance(container, dict) or isinstance(container, Mapping)):
                continue
            for text_key in ("content", "refusal", "reasoning_content", "reasoning"):
                val = container.get(text_key)
                if isinstance(val, str) and val:
                    parts.append(val)
            func_call = container.get("function_call")
            if isinstance(func_call, dict) or isinstance(func_call, Mapping):
                for fk in ("name", "arguments"):
                    fval = func_call.get(fk)
                    if isinstance(fval, str) and fval:
                        parts.append(fval)
            tool_calls = container.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not (isinstance(tc, dict) or isinstance(tc, Mapping)):
                        continue
                    fn = tc.get("function")
                    if isinstance(fn, dict) or isinstance(fn, Mapping):
                        for fk in ("name", "arguments"):
                            fval = fn.get(fk)
                            if isinstance(fval, str) and fval:
                                parts.append(fval)
            audio = container.get("audio")
            if isinstance(audio, dict) or isinstance(audio, Mapping):
                transcript = audio.get("transcript")
                if isinstance(transcript, str) and transcript:
                    parts.append(transcript)
    return "".join(parts)


def _has_choice_audio_output(obj: Mapping[str, Any]) -> bool:
    """Return True when any choice carries generated audio data or transcript."""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not (isinstance(choice, dict) or isinstance(choice, Mapping)):
            continue
        for container_key in ("delta", "message"):
            container = choice.get(container_key)
            if not (isinstance(container, dict) or isinstance(container, Mapping)):
                continue
            audio = container.get("audio")
            if isinstance(audio, dict) or isinstance(audio, Mapping):
                for ak in ("data", "transcript"):
                    aval = audio.get(ak)
                    if isinstance(aval, str) and aval:
                        return True
    return False


_MAX_SAFE_INT_DIGITS = 18


def _safe_nonneg_int(val: Any) -> Optional[int]:
    """Parse a non-negative integer count without raising on malformed telemetry."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if val >= 0 else None
    if isinstance(val, float):
        if math.isfinite(val) and val >= 0.0 and val.is_integer():
            try:
                return int(val)
            except (ValueError, OverflowError):
                return None
        return None
    if isinstance(val, str):
        cleaned = val.strip()
        if (
            cleaned
            and len(cleaned) <= _MAX_SAFE_INT_DIGITS
            and cleaned.isascii()
            and cleaned.isdigit()
        ):
            try:
                return int(cleaned)
            except (ValueError, OverflowError):
                return None
        return None
    return None


def _extract_non_stream_usage(
    raw_resp: bytes,
    est_prompt_tokens: int,
    *,
    parsed_resp: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, int]:
    """Extract `(prompt_tokens, completion_tokens)` preferring upstream counts."""
    prompt_tokens = est_prompt_tokens
    completion_tokens = 0
    saw_prompt = False
    saw_completion = False
    if parsed_resp is None:
        try:
            loaded = json.loads(raw_resp.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return prompt_tokens, completion_tokens
        if not isinstance(loaded, Mapping):
            return prompt_tokens, completion_tokens
        parsed_resp = loaded

    usage = parsed_resp.get("usage")
    if isinstance(usage, Mapping):
        pt = _safe_nonneg_int(usage.get("prompt_tokens"))
        if pt is not None:
            prompt_tokens = pt
            saw_prompt = True
        ct = _safe_nonneg_int(usage.get("completion_tokens"))
        if ct is not None:
            completion_tokens = ct
            saw_completion = True

    timings = parsed_resp.get("timings")
    if isinstance(timings, Mapping):
        if not saw_prompt:
            pt = _safe_nonneg_int(timings.get("prompt_n"))
            if pt is not None:
                prompt_tokens = pt
                saw_prompt = True
        if not saw_completion:
            ct = _safe_nonneg_int(timings.get("predicted_n"))
            if ct is not None:
                completion_tokens = ct
                saw_completion = True

    if not saw_completion:
        output_txt = _extract_choice_output_text(parsed_resp)
        if output_txt:
            completion_tokens = max(1, (len(output_txt) + 3) // 4)

    return prompt_tokens, completion_tokens


_NON_RETRYABLE_SSE_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "bad_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "unprocessable_entity_error",
    }
)
_NON_RETRYABLE_SSE_ERROR_CODES = frozenset(
    {
        "context_length_exceeded",
        "invalid_request_error",
        "bad_request",
        "model_not_found",
        "invalid_api_key",
    }
)


def _classify_sse_error_event(
    event_type: str,
    payload_bytes: bytes,
) -> Optional[Tuple[bool, int, bytes]]:
    """Classify an upstream SSE error event before committing a 200 stream.

    Returns `None` if the event is not an error event, otherwise returns
    `(is_retryable, http_status, json_body_bytes)`.
    """
    is_event_err = event_type.strip().lower() == "error"
    parsed: Any = None
    if payload_bytes and payload_bytes != b"[DONE]":
        try:
            parsed = json.loads(payload_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            parsed = None

    has_err_obj = isinstance(parsed, Mapping) and (
        parsed.get("error") is not None or parsed.get("object") == "error"
    )
    if not is_event_err and not has_err_obj:
        return None

    err_mapping: Mapping[str, Any] = {}
    if isinstance(parsed, Mapping):
        inner = parsed.get("error")
        if isinstance(inner, Mapping):
            err_mapping = inner
        else:
            err_mapping = parsed

    raw_status: Optional[int] = None
    for candidate_map in (err_mapping, parsed if isinstance(parsed, Mapping) else {}):
        for key in ("status", "http_status", "status_code", "code"):
            val = candidate_map.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and 400 <= val <= 599:
                raw_status = val
                break
        if raw_status is not None:
            break

    err_type = str(err_mapping.get("type") or "").strip().lower()
    err_code = str(err_mapping.get("code") or "").strip().lower()
    is_client_error = (
        (raw_status is not None and 400 <= raw_status < 500 and raw_status != 429)
        or err_type in _NON_RETRYABLE_SSE_ERROR_TYPES
        or err_code in _NON_RETRYABLE_SSE_ERROR_CODES
    )

    if is_client_error:
        status_code = raw_status if (raw_status is not None and 400 <= raw_status < 500) else 400
        is_retryable = False
    else:
        status_code = (
            raw_status
            if (raw_status is not None and (500 <= raw_status <= 599 or raw_status == 429))
            else 502
        )
        is_retryable = True

    if isinstance(parsed, Mapping) and parsed.get("error") is not None:
        body = (json.dumps(parsed) + "\n").encode("utf-8")
    elif isinstance(parsed, Mapping):
        msg = str(parsed.get("message") or "Upstream SSE error")
        body = _openai_error_body(
            msg,
            error_type=err_type or ("invalid_request_error" if is_client_error else "server_error"),
            code=err_code or None,
        )
    else:
        raw_msg = payload_bytes.decode("utf-8", errors="replace").strip() or "Upstream SSE error"
        body = _openai_error_body(raw_msg, error_type="server_error", code="upstream_sse_error")

    return is_retryable, status_code, body


def _sse_event_should_commit(payload_bytes: bytes) -> bool:
    """Return True when a complete non-error SSE data event carries output or a terminator.

    Role-only, empty-delta, and prompt-progress chunks are buffered so Spot
    preemption before the first generated token still fails over. Raises
    `_MalformedSSEPayloadError` on invalid UTF-8, invalid JSON, or non-object
    JSON so pre-commit protocol corruption remains eligible for failover.
    """
    if not payload_bytes:
        return False
    if payload_bytes == b"[DONE]":
        return True
    try:
        obj = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _MalformedSSEPayloadError("Malformed JSON in upstream SSE event") from exc
    if not isinstance(obj, Mapping):
        raise _MalformedSSEPayloadError("Upstream SSE event JSON must be an object")
    if obj.get("error") is not None or obj.get("object") == "error":
        return False
    if _extract_choice_output_text(obj) or _has_choice_audio_output(obj):
        return True
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                return True
        if not choices and isinstance(obj.get("usage"), Mapping):
            return True
        return False
    if "prompt_progress" in obj:
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
            resolved_auth = load_hook_from_spec(
                spec.strip(),
                auth_cfg.get("kwargs"),
                role="auth",
            )
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
            resolved_usage = load_hook_from_spec(
                str(spec),
                usage_cfg.get("kwargs"),
                role="usage_meter",
            )
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
            resolved_scaler = load_hook_from_spec(
                str(spec),
                scaler_cfg.get("kwargs"),
                role="scaler",
            )
        else:
            resolved_scaler = NoOpScalerHook()

    validate_hook_for_role(resolved_auth, "auth")
    validate_hook_for_role(resolved_usage, "usage_meter")
    validate_hook_for_role(resolved_scaler, "scaler")
    return resolved_auth, resolved_usage, resolved_scaler


class _BackendSlotLease:
    """RAII lease guaranteeing an acquired backend in-flight slot is released once."""

    __slots__ = ("_engine", "_state", "_released")

    def __init__(self, engine: "RouterEngine", state: BackendState) -> None:
        self._engine = engine
        self._state = state
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._engine.release_backend(self._state)

    def __enter__(self) -> "_BackendSlotLease":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def __del__(self) -> None:
        if not getattr(self, "_released", True):
            try:
                self.release()
            except Exception:
                pass


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
        self._rr_cursors: OrderedDict[bytes, int] = OrderedDict()
        self._metric_models_seen: set[str] = set()
        self._req_mem_lock = threading.Lock()
        self._req_mem_used_bytes = 0
        self._resp_mem_lock = threading.Lock()
        self._resp_mem_used_bytes = 0
        self._poller_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._usage_err_log_lock = threading.Lock()
        self._usage_err_last_log_monotonic: Optional[float] = None
        self._usage_err_suppressed = 0

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

    def try_reserve_request_memory(self, amount_bytes: int) -> bool:
        """Atomically reserve bytes from the shared in-flight request memory budget."""
        need = max(0, amount_bytes)
        limit = self.config.max_total_request_memory_bytes
        with self._req_mem_lock:
            if self._req_mem_used_bytes + need > limit:
                return False
            self._req_mem_used_bytes += need
            return True

    def release_request_memory(self, amount_bytes: int) -> None:
        """Release previously reserved bytes back to the shared request memory budget."""
        rel = max(0, amount_bytes)
        if rel == 0:
            return
        with self._req_mem_lock:
            self._req_mem_used_bytes = max(0, self._req_mem_used_bytes - rel)

    def try_reserve_response_memory(self, amount_bytes: int) -> bool:
        """Atomically reserve bytes from the shared in-flight response memory budget."""
        need = max(0, amount_bytes)
        limit = self.config.max_total_response_memory_bytes
        with self._resp_mem_lock:
            if self._resp_mem_used_bytes + need > limit:
                return False
            self._resp_mem_used_bytes += need
            return True

    def available_response_memory(self) -> int:
        """Return currently unreserved bytes in the shared response memory budget."""
        with self._resp_mem_lock:
            return max(0, self.config.max_total_response_memory_bytes - self._resp_mem_used_bytes)

    def release_response_memory(self, amount_bytes: int) -> None:
        """Release previously reserved bytes back to the shared response memory budget."""
        rel = max(0, amount_bytes)
        if rel == 0:
            return
        with self._resp_mem_lock:
            self._resp_mem_used_bytes = max(0, self._resp_mem_used_bytes - rel)

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
                    state.unhealthy_reason = _safe_probe_error(exc)

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
        *,
        strip_api_version: bool = False,
    ) -> Tuple[int, Optional[Any], float, Optional[str]]:
        scheme, host, port, path = _join_backend_url(
            backend.base_url,
            endpoint,
            strip_api_version=strip_api_version,
        )
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "Connection": "close",
        }
        t0 = time.monotonic()
        deadline = t0 + max(0.01, timeout_s)
        conn: Optional[http.client.HTTPConnection] = None
        resp: Optional[http.client.HTTPResponse] = None
        raw_sock: Any = None
        try:
            api_key = backend.resolved_api_key()
            _ensure_safe_credential_transport(backend, scheme, host, api_key)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError("Probe exceeded elapsed deadline")
            conn = _open_http_connection(scheme, host, port, rem)
            conn.request("GET", path, headers=headers)
            raw_sock = conn.sock
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError("Probe exceeded elapsed deadline")
            deadline_sock: Optional[DeadlineSocket] = None
            if raw_sock is not None:
                deadline_sock = DeadlineSocket(raw_sock, deadline, label="Probe response")
                conn.sock = deadline_sock
            resp = conn.getresponse()
            upstream_sock = deadline_sock if deadline_sock is not None else raw_sock
            raw = _read_bounded_http_response(
                resp,
                upstream_sock,
                max_bytes=MAX_PROBE_RESPONSE_BYTES,
                deadline=deadline,
                label="Probe response",
                chunk_size=8192,
            )
            rtt_ms = (time.monotonic() - t0) * 1000.0
            parsed_json: Optional[Any] = None
            if raw:
                try:
                    loaded = json.loads(raw.decode("utf-8", errors="replace"))
                    if isinstance(loaded, (dict, list)):
                        parsed_json = loaded
                except json.JSONDecodeError:
                    parsed_json = None
            return resp.status, parsed_json, rtt_ms, None
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
            rtt_ms = (time.monotonic() - t0) * 1000.0
            return 0, None, rtt_ms, _safe_probe_error(exc)
        finally:
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
            if raw_sock is not None:
                try:
                    raw_sock.close()
                except OSError:
                    pass

    def probe_backend(self, state: BackendState) -> None:
        """Probe backend HTTP readiness, `/v1/models`, and optional `tpu-exporter`."""
        probe_deadline = time.monotonic() + self.config.probe_timeout_s
        probe_ok = False
        probe_err: Optional[str] = None
        rtt_ms: Optional[float] = None
        discovered: Optional[List[str]] = None

        def remaining_timeout() -> float:
            return probe_deadline - time.monotonic()

        if state.config.health_path:
            rem = remaining_timeout()
            if rem <= 0:
                status, err = 0, "probe_deadline_exceeded"
            else:
                status, _, rtt_ms, err = self._http_get_json(
                    state.config,
                    state.config.health_path,
                    rem,
                    strip_api_version=True,
                )
            if 200 <= status < 300:
                probe_ok = True
            else:
                probe_err = err or f"health_http_{status}"
            if probe_ok and state.config.discover_models:
                rem = remaining_timeout()
                if rem > 0:
                    m_status, m_json, _, _ = self._http_get_json(
                        state.config, "/v1/models", rem
                    )
                    if 200 <= m_status < 300:
                        models_list = _extract_models_list(m_json)
                        if models_list is not None:
                            discovered = models_list
        else:
            rem = remaining_timeout()
            if rem <= 0:
                status, m_json, err = 0, None, "probe_deadline_exceeded"
            else:
                status, m_json, rtt_ms, err = self._http_get_json(
                    state.config, "/v1/models", rem
                )
            if 200 <= status < 300:
                models_list = _extract_models_list(m_json)
                if models_list is not None:
                    probe_ok = True
                    discovered = models_list
                else:
                    probe_ok = False
                    probe_err = "invalid_models_response"
            else:
                probe_err = err or f"probe_http_{status}"

        tpu_tele: Optional[TpuExporterTelemetry] = None
        tpu_ready = True
        tpu_reason: Optional[str] = None
        if state.config.tpu_exporter_url:
            rem = remaining_timeout()
            if rem <= 0:
                tpu_tele = TpuExporterTelemetry(
                    reachable=False,
                    exporter_up=False,
                    devices_discovered=0,
                    healthy_chips=0,
                    total_chips=0,
                    error="Probe elapsed deadline exceeded before exporter scrape",
                )
            else:
                tpu_tele = scrape_tpu_exporter(
                    state.config.tpu_exporter_url, timeout_s=rem
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
        try:
            return self.metrics.inc_inflight(state.config.id)
        except BaseException:
            with self._lock:
                state.inflight = max(0, state.inflight - 1)
            raise

    def reserve_backend_lease(
        self,
        state: BackendState,
        *,
        now_monotonic: Optional[float] = None,
    ) -> Optional[Tuple[_BackendSlotLease, int]]:
        """Reserve an in-flight slot and return `(lease, total_inflight)`."""
        reserved = self.try_reserve_backend(state, now_monotonic=now_monotonic)
        if reserved is None:
            return None
        return _BackendSlotLease(self, state), reserved

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

            # round_robin (bounded per-model LRU with fixed 32-byte SHA-256 keys)
            eligible.sort(key=lambda b: (b.config.priority, b.config.id))
            rr_key = hashlib.sha256(model.encode("utf-8", errors="replace")).digest()
            cursor = self._rr_cursors.get(rr_key, 0)
            if not excluded:
                self._rr_cursors[rr_key] = (cursor + 1) % len(eligible)
                self._rr_cursors.move_to_end(rr_key)
                while len(self._rr_cursors) > MAX_RR_CURSORS:
                    self._rr_cursors.popitem(last=False)
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

    def health_payload(self, *, detailed: bool = True) -> Tuple[int, Dict[str, Any]]:
        now_m = time.monotonic()
        with self._lock:
            snaps = [b.snapshot(now_m) for b in self._backends]
        healthy_count = sum(1 for s in snaps if s["enabled"] and s["healthy"])
        status_str = "ok" if healthy_count > 0 else "degraded"
        http_code = 200 if healthy_count > 0 else 503
        if not detailed:
            return http_code, {"status": status_str}
        enabled_count = sum(1 for s in snaps if s["enabled"])
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

    def _on_usage_hook_error(self, exc: Exception) -> None:
        total_errors = self.metrics.record_usage_meter_error()
        if isinstance(self.usage_hook, JsonlUsageMeterHook):
            # JsonlUsageMeterHook already emitted its own rate-limited stderr warning.
            return
        now = time.monotonic()
        with self._usage_err_log_lock:
            if (
                self._usage_err_last_log_monotonic is None
                or (now - self._usage_err_last_log_monotonic) >= 30.0
            ):
                suppressed = self._usage_err_suppressed
                self._usage_err_suppressed = 0
                self._usage_err_last_log_monotonic = now
                suffix = (
                    f" ({suppressed} suppressed since last warning)"
                    if suppressed > 0
                    else ""
                )
                try:
                    print(
                        f"[freetpu-router] WARNING: usage meter hook failed: "
                        f"{type(exc).__name__}: {exc} "
                        f"(total_usage_meter_errors={total_errors}){suffix}",
                        file=sys.stderr,
                    )
                except OSError:
                    pass
            else:
                self._usage_err_suppressed += 1

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
            model=model[:MAX_MODEL_ID_LEN],
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
        except Exception as exc:
            self._on_usage_hook_error(exc)

    def _record_terminal_failure(
        self,
        *,
        request_id: str,
        model: str,
        backend_id: str,
        matched_backend: bool,
        status_code: int,
        latency_ms: float,
        stream: bool,
        failovers: int,
        principal: str,
    ) -> None:
        metric_model = self.metric_model_label(model, matched_backend=matched_backend)
        self.metrics.record_request(
            model=metric_model,
            backend_id=backend_id,
            status=status_code,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            ttft_ms=None,
            estimated_cost_usd=0.0,
        )
        event = UsageEvent(
            request_id=request_id,
            model=model[:MAX_MODEL_ID_LEN],
            backend_id=backend_id,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=round(latency_ms, 3),
            estimated_cost_usd=0.0,
            ttft_ms=None,
            stream=stream,
            failovers=failovers,
            principal=principal,
            status_code=status_code,
        )
        try:
            self.usage_hook.record_usage(event)
        except Exception as exc:
            self._on_usage_hook_error(exc)


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
    """Buffered reader wrapper enforcing a wall-clock deadline and cumulative header byte cap."""

    def __init__(
        self,
        raw_rfile: Any,
        sock: Any,
        deadline: float,
        *,
        max_header_bytes: int,
    ) -> None:
        self._raw = raw_rfile
        self._sock = sock
        self._deadline = deadline
        self._max_header_bytes = max_header_bytes
        self._header_bytes_read = 0
        self._buf = bytearray()

    def readline(self, limit: int = -1) -> bytes:
        max_len = limit if limit is not None and limit >= 0 else None
        read_fn = getattr(self._raw, "read1", self._raw.read)
        while True:
            budget_left = self._max_header_bytes - self._header_bytes_read
            if budget_left <= 0:
                raise _HeaderSizeLimitExceeded("Request headers exceeded maximum byte limit")
            nl_idx = self._buf.find(b"\n")
            if nl_idx != -1:
                end = nl_idx + 1
                if max_len is not None and end > max_len:
                    end = max_len
                if end > budget_left:
                    raise _HeaderSizeLimitExceeded(
                        "Request headers exceeded maximum byte limit"
                    )
                out = bytes(self._buf[:end])
                del self._buf[:end]
                self._header_bytes_read += len(out)
                return out
            if len(self._buf) > budget_left:
                raise _HeaderSizeLimitExceeded("Request headers exceeded maximum byte limit")
            if max_len is not None and len(self._buf) >= max_len:
                if max_len > budget_left:
                    raise _HeaderSizeLimitExceeded(
                        "Request headers exceeded maximum byte limit"
                    )
                out = bytes(self._buf[:max_len])
                del self._buf[:max_len]
                self._header_bytes_read += len(out)
                return out
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Client header read timed out")
            self._sock.settimeout(remaining)
            to_read = min(4096, (budget_left - len(self._buf)) + 1)
            chunk = read_fn(to_read)
            if not chunk:
                if len(self._buf) > budget_left:
                    raise _HeaderSizeLimitExceeded(
                        "Request headers exceeded maximum byte limit"
                    )
                out = bytes(self._buf)
                self._buf.clear()
                self._header_bytes_read += len(out)
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


_MSG_DONTWAIT = getattr(socket, "MSG_DONTWAIT", 0)
_SSE_CLIENT_WRITE_BUFFER_MIN_BYTES = 256 * 1024
_SSE_CLIENT_WRITE_BUFFER_MAX_BYTES = 2 * 1024 * 1024
_SSE_REACTOR_MAX_READS_PER_TICK = 16


class _SSEReactorTask:
    """State for a committed plain-TCP SSE stream pumped by `_SSEReactor`."""

    __slots__ = (
        "upstream_sock",
        "client_sock",
        "engine",
        "sse_buf",
        "raw_buf",
        "write_buf",
        "chunked",
        "chunked_downstream",
        "chunk_left",
        "remaining_content_length",
        "max_event_bytes",
        "max_write_buf_bytes",
        "timeout_s",
        "write_timeout_s",
        "progress_deadline",
        "write_deadline",
        "sse_mem_box",
        "on_event_fn",
        "event_data_lines",
        "current_event_type",
        "scan_pos",
        "event_start",
        "skip_leading_lf",
        "saw_done_sentinel",
        "upstream_read_error",
        "downstream_write_error",
        "size_limit_exceeded",
        "memory_limit_exceeded",
        "eof",
        "upstream_done",
        "done_event",
        "upstream_registered",
        "watch_client_read",
        "client_events",
        "client_orig_timeout",
        "client_nonblocking_set",
    )

    def __init__(
        self,
        *,
        upstream_sock: socket.socket,
        client_sock: socket.socket,
        engine: RouterEngine,
        sse_buf: bytearray,
        raw_buf: bytearray,
        chunked: bool,
        chunk_left: Optional[int],
        remaining_content_length: Optional[int],
        max_event_bytes: int,
        timeout_s: float,
        progress_deadline: float,
        sse_mem_box: List[int],
        on_event_fn: Callable[[str, List[bytes]], Tuple[bool, bool]],
        event_data_lines: List[bytes],
        current_event_type: str,
        chunked_downstream: bool = True,
        skip_leading_lf: bool = False,
        max_write_buf_bytes: Optional[int] = None,
        write_timeout_s: Optional[float] = None,
    ) -> None:
        self.upstream_sock = upstream_sock
        self.client_sock = client_sock
        self.engine = engine
        self.sse_buf = sse_buf
        self.raw_buf = raw_buf
        self.write_buf = bytearray()
        self.chunked = chunked
        self.chunked_downstream = chunked_downstream
        self.chunk_left = chunk_left
        self.remaining_content_length = remaining_content_length
        self.max_event_bytes = max_event_bytes
        if max_write_buf_bytes is not None:
            self.max_write_buf_bytes = max(1, max_write_buf_bytes)
        else:
            self.max_write_buf_bytes = max(
                _SSE_CLIENT_WRITE_BUFFER_MIN_BYTES,
                min(_SSE_CLIENT_WRITE_BUFFER_MAX_BYTES, max_event_bytes + 64),
            )
        self.timeout_s = timeout_s
        if write_timeout_s is not None:
            self.write_timeout_s = max(0.01, write_timeout_s)
        else:
            self.write_timeout_s = max(
                0.01,
                min(timeout_s, engine.config.client_read_timeout_s),
            )
        self.progress_deadline = progress_deadline
        self.write_deadline: Optional[float] = None
        self.sse_mem_box = sse_mem_box
        self.on_event_fn = on_event_fn
        self.event_data_lines = event_data_lines
        self.current_event_type = current_event_type
        self.scan_pos = 0
        self.event_start = 0
        self.skip_leading_lf = skip_leading_lf
        self.saw_done_sentinel = False
        self.upstream_read_error = False
        self.downstream_write_error = False
        self.size_limit_exceeded = False
        self.memory_limit_exceeded = False
        self.eof = False
        self.upstream_done = False
        self.done_event = threading.Event()
        self.upstream_registered = False
        self.watch_client_read = True
        self.client_events = 0
        self.client_orig_timeout: Optional[float] = None
        self.client_nonblocking_set = False


class _SSEReactor:
    """Single-thread `selectors` event loop pumping committed SSE streams at scale."""

    def __init__(self) -> None:
        self._selector = selectors.DefaultSelector()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._selector.register(self._wake_r, selectors.EVENT_READ, data=None)
        self._pending_lock = threading.Lock()
        self._pending: deque[_SSEReactorTask] = deque()
        self._active: set[_SSEReactorTask] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="freetpu-sse-reactor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._wake_w.send(b"\x00")
        except OSError:
            pass
        thread: Optional[threading.Thread]
        with self._start_lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        for sock in (self._wake_r, self._wake_w):
            try:
                self._selector.unregister(sock)
            except (KeyError, ValueError):
                pass
            try:
                sock.close()
            except OSError:
                pass
        try:
            self._selector.close()
        except OSError:
            pass

    def submit(self, task: _SSEReactorTask) -> None:
        if self._stop_event.is_set():
            task.downstream_write_error = True
            task.done_event.set()
            return
        self.start()
        with self._pending_lock:
            self._pending.append(task)
        try:
            self._wake_w.send(b"\x01")
        except BlockingIOError:
            pass
        except OSError:
            pass

    def _release_mem(self, task: _SSEReactorTask, amount: int) -> None:
        if amount <= 0:
            return
        rel = min(amount, task.sse_mem_box[0])
        if rel > 0:
            task.engine.release_response_memory(rel)
            task.sse_mem_box[0] -= rel

    def _set_upstream_reading(self, task: _SSEReactorTask, enabled: bool) -> bool:
        if enabled == task.upstream_registered:
            return True
        try:
            if enabled:
                self._selector.register(
                    task.upstream_sock, selectors.EVENT_READ, data=("up", task)
                )
                task.upstream_registered = True
            else:
                self._selector.unregister(task.upstream_sock)
                task.upstream_registered = False
            return True
        except (KeyError, OSError, ValueError):
            task.upstream_registered = False
            if enabled:
                task.upstream_read_error = True
                self._finish_task(task)
                return False
            return True

    def _update_client_registration(self, task: _SSEReactorTask) -> bool:
        desired = 0
        if task.watch_client_read:
            desired |= selectors.EVENT_READ
        if task.write_buf:
            desired |= selectors.EVENT_WRITE
        if desired == task.client_events:
            return True
        try:
            if desired == 0:
                if task.client_events != 0:
                    self._selector.unregister(task.client_sock)
            elif task.client_events == 0:
                self._selector.register(task.client_sock, desired, data=("cli", task))
            else:
                self._selector.modify(task.client_sock, desired, data=("cli", task))
            task.client_events = desired
            return True
        except (KeyError, OSError, ValueError):
            task.client_events = 0
            task.downstream_write_error = True
            self._finish_task(task)
            return False

    def _finish_task(self, task: _SSEReactorTask) -> None:
        self._active.discard(task)
        if task.upstream_registered:
            try:
                self._selector.unregister(task.upstream_sock)
            except (KeyError, ValueError):
                pass
            task.upstream_registered = False
        if task.client_events != 0:
            try:
                self._selector.unregister(task.client_sock)
            except (KeyError, ValueError):
                pass
            task.client_events = 0
        if task.client_nonblocking_set:
            try:
                task.client_sock.settimeout(task.client_orig_timeout)
            except OSError:
                pass
            task.client_nonblocking_set = False
        if task.write_buf:
            rem = len(task.write_buf)
            task.write_buf.clear()
            self._release_mem(task, rem)
        if task.sse_buf:
            rem = len(task.sse_buf)
            task.sse_buf.clear()
            self._release_mem(task, rem)
        if task.raw_buf:
            rem = len(task.raw_buf)
            task.raw_buf.clear()
            self._release_mem(task, rem)
        task.done_event.set()

    def _enqueue_client_write(self, task: _SSEReactorTask, framed: bytes) -> bool:
        """Write `framed` nonblockingly to `task.client_sock`, buffering any remainder.

        Returns False on downstream write error or buffer/memory cap overflow.
        """
        if not framed:
            return True
        remaining: bytes
        if not task.write_buf:
            try:
                sent = task.client_sock.send(framed)
            except BlockingIOError:
                remaining = framed
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    remaining = framed
                else:
                    task.downstream_write_error = True
                    return False
            else:
                if sent <= 0:
                    task.downstream_write_error = True
                    return False
                if sent >= len(framed):
                    return True
                remaining = framed[sent:]
        else:
            remaining = framed

        rem_len = len(remaining)
        if len(task.write_buf) + rem_len > task.max_write_buf_bytes:
            task.downstream_write_error = True
            return False
        if not task.engine.try_reserve_response_memory(rem_len):
            task.downstream_write_error = True
            return False
        task.sse_mem_box[0] += rem_len
        was_empty = not task.write_buf
        task.write_buf.extend(remaining)
        if was_empty:
            task.write_deadline = time.monotonic() + task.write_timeout_s
        return True

    def _decode_chunked(self, task: _SSEReactorTask) -> None:
        raw = task.raw_buf
        while raw:
            if task.chunk_left is None:
                nl = raw.find(b"\n")
                if nl == -1:
                    if len(raw) > 4096:
                        task.upstream_read_error = True
                    return
                line = bytes(raw[: nl + 1])
                del raw[: nl + 1]
                self._release_mem(task, len(line))
                semi = line.find(b";")
                size_bytes = line[:semi] if semi != -1 else line
                try:
                    size = int(size_bytes.strip(), 16)
                except ValueError:
                    task.upstream_read_error = True
                    return
                if size < 0:
                    task.upstream_read_error = True
                    return
                if size == 0:
                    task.eof = True
                    return
                task.chunk_left = size
            if task.chunk_left > 0:
                take = min(task.chunk_left, len(raw))
                if take == 0:
                    return
                task.sse_buf.extend(raw[:take])
                del raw[:take]
                task.chunk_left -= take
            if task.chunk_left == 0:
                if not raw:
                    return
                if raw[0] == 13:
                    if len(raw) < 2:
                        return
                    if raw[1] == 10:
                        del raw[:2]
                        self._release_mem(task, 2)
                    else:
                        task.upstream_read_error = True
                        return
                elif raw[0] == 10:
                    del raw[:1]
                    self._release_mem(task, 1)
                else:
                    task.upstream_read_error = True
                    return
                task.chunk_left = None

    def _process_sse_buf(self, task: _SSEReactorTask) -> None:
        """Scan complete SSE events in `task.sse_buf` and write nonblockingly as one HTTP chunk."""
        buf = task.sse_buf
        flush_end = 0
        while True:
            if task.skip_leading_lf:
                if task.scan_pos >= len(buf):
                    break
                task.skip_leading_lf = False
                if buf[task.scan_pos] == 10:
                    was_event_start = task.event_start == task.scan_pos
                    task.scan_pos += 1
                    if was_event_start:
                        task.event_start = task.scan_pos
                        flush_end = task.scan_pos
                    continue
            nl_idx = buf.find(b"\n", task.scan_pos)
            cr_idx = buf.find(b"\r", task.scan_pos)
            if nl_idx == -1 and cr_idx == -1:
                if len(buf) - task.event_start > task.max_event_bytes:
                    task.size_limit_exceeded = True
                break
            if cr_idx != -1 and (nl_idx == -1 or cr_idx < nl_idx):
                content_end = cr_idx
                if cr_idx + 1 < len(buf) and buf[cr_idx + 1] == 10:
                    line_end = cr_idx + 2
                else:
                    line_end = cr_idx + 1
                    if line_end == len(buf):
                        task.skip_leading_lf = True
            else:
                content_end = nl_idx
                line_end = nl_idx + 1
            if line_end - task.event_start > task.max_event_bytes:
                task.size_limit_exceeded = True
                break
            if content_end == task.scan_pos:
                ev_type = task.current_event_type
                task.current_event_type = ""
                made_progress, is_done = task.on_event_fn(ev_type, task.event_data_lines)
                task.event_data_lines.clear()
                if is_done:
                    task.saw_done_sentinel = True
                if made_progress:
                    task.progress_deadline = time.monotonic() + task.timeout_s
                task.event_start = line_end
                flush_end = line_end
                task.scan_pos = line_end
                if task.saw_done_sentinel:
                    break
            else:
                if buf.startswith(b"data:", task.scan_pos):
                    data_start = task.scan_pos + 5
                    if data_start < content_end and buf[data_start] == 32:
                        data_start += 1
                    task.event_data_lines.append(bytes(buf[data_start:content_end]))
                elif buf.startswith(b"event:", task.scan_pos):
                    task.current_event_type = (
                        bytes(buf[task.scan_pos + 6 : content_end])
                        .strip()
                        .decode("utf-8", errors="replace")
                    )
                task.scan_pos = line_end

        if flush_end > 0:
            payload = bytes(buf[:flush_end])
            del buf[:flush_end]
            task.scan_pos -= flush_end
            task.event_start -= flush_end
            self._release_mem(task, flush_end)
            framed = (
                f"{flush_end:x}\r\n".encode("ascii") + payload + b"\r\n"
                if task.chunked_downstream
                else payload
            )
            if not self._enqueue_client_write(task, framed):
                return

        if (
            task.saw_done_sentinel
            or task.size_limit_exceeded
            or task.upstream_read_error
            or task.memory_limit_exceeded
            or task.eof
        ):
            task.upstream_done = True

    def _on_upstream_ready(self, task: _SSEReactorTask) -> None:
        reads = 0
        while reads < _SSE_REACTOR_MAX_READS_PER_TICK:
            if task.write_buf or task.upstream_done:
                return
            avail = task.engine.available_response_memory()
            rem_ev_cap = (task.max_event_bytes - (len(task.sse_buf) - task.event_start)) + 1
            to_read = min(65536, max(1, avail), max(1, rem_ev_cap))
            if not task.chunked and task.remaining_content_length is not None:
                if task.remaining_content_length <= 0:
                    task.eof = True
                    task.upstream_done = True
                    self._set_upstream_reading(task, False)
                    if not task.write_buf:
                        self._finish_task(task)
                    else:
                        self._update_client_registration(task)
                    return
                to_read = min(to_read, task.remaining_content_length)
            try:
                chunk = task.upstream_sock.recv(to_read)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return
                task.upstream_read_error = True
                task.upstream_done = True
                self._set_upstream_reading(task, False)
                if not task.write_buf:
                    self._finish_task(task)
                else:
                    self._update_client_registration(task)
                return

            reads += 1
            if not chunk:
                if task.chunked:
                    task.upstream_read_error = True
                elif (
                    task.remaining_content_length is not None
                    and task.remaining_content_length > 0
                ):
                    task.upstream_read_error = True
                task.eof = True
                task.upstream_done = True
                self._process_sse_buf(task)
                self._set_upstream_reading(task, False)
                if task.downstream_write_error or not task.write_buf:
                    self._finish_task(task)
                else:
                    self._update_client_registration(task)
                return

            if not task.engine.try_reserve_response_memory(len(chunk)):
                task.memory_limit_exceeded = True
                task.upstream_done = True
                self._set_upstream_reading(task, False)
                if not task.write_buf:
                    self._finish_task(task)
                else:
                    self._update_client_registration(task)
                return
            task.sse_mem_box[0] += len(chunk)

            if task.chunked:
                task.raw_buf.extend(chunk)
                self._decode_chunked(task)
            else:
                if task.remaining_content_length is not None:
                    task.remaining_content_length = max(
                        0, task.remaining_content_length - len(chunk)
                    )
                    if task.remaining_content_length == 0:
                        task.eof = True
                task.sse_buf.extend(chunk)

            self._process_sse_buf(task)
            if task.downstream_write_error:
                self._finish_task(task)
                return
            if task.upstream_done:
                self._set_upstream_reading(task, False)
                if not task.write_buf:
                    self._finish_task(task)
                else:
                    self._update_client_registration(task)
                return
            if task.write_buf:
                # Apply backpressure: pause upstream reads until client drains write_buf.
                self._set_upstream_reading(task, False)
                self._update_client_registration(task)
                return

    def _on_client_writable(self, task: _SSEReactorTask) -> None:
        total_sent = 0
        while task.write_buf:
            view = memoryview(task.write_buf)[:65536]
            try:
                sent = task.client_sock.send(view)
            except BlockingIOError:
                view.release()
                break
            except OSError as exc:
                view.release()
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                task.downstream_write_error = True
                self._finish_task(task)
                return
            view.release()
            if sent <= 0:
                task.downstream_write_error = True
                self._finish_task(task)
                return
            del task.write_buf[:sent]
            self._release_mem(task, sent)
            total_sent += sent

        if not task.write_buf:
            task.write_deadline = None
            if task.upstream_done:
                self._finish_task(task)
                return
            task.progress_deadline = time.monotonic() + task.timeout_s
            if not self._update_client_registration(task):
                return
            self._set_upstream_reading(task, True)
        elif total_sent > 0:
            task.write_deadline = time.monotonic() + task.write_timeout_s

    def _on_client_readable(self, task: _SSEReactorTask) -> None:
        try:
            task.client_sock.recv(1, socket.MSG_PEEK | _MSG_DONTWAIT)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            task.downstream_write_error = True
            self._finish_task(task)
            return
        # Orderly read-side EOF (client half-closed via SHUT_WR per RFC 9112 §9.6)
        # or extra pipelined bytes: stop watching client readability while
        # preserving EVENT_WRITE and keeping the response stream open.
        task.watch_client_read = False
        self._update_client_registration(task)

    def _drain_pending(self) -> None:
        while True:
            with self._pending_lock:
                if not self._pending:
                    return
                task = self._pending.popleft()
            try:
                try:
                    task.client_orig_timeout = task.client_sock.gettimeout()
                    task.client_sock.setblocking(False)
                    task.client_nonblocking_set = True
                except OSError:
                    task.downstream_write_error = True
                    self._finish_task(task)
                    continue
                task.upstream_sock.setblocking(False)
                if task.chunked and task.raw_buf:
                    self._decode_chunked(task)
                elif not task.chunked and task.remaining_content_length == 0:
                    task.eof = True
                self._process_sse_buf(task)
                if task.downstream_write_error or (
                    task.upstream_done and not task.write_buf
                ):
                    self._finish_task(task)
                    continue
                self._active.add(task)
                if not self._update_client_registration(task):
                    continue
                if not task.upstream_done and not task.write_buf:
                    self._set_upstream_reading(task, True)
            except Exception:
                task.upstream_read_error = True
                self._finish_task(task)

    @staticmethod
    def _task_deadline(task: _SSEReactorTask) -> float:
        if task.write_buf and task.write_deadline is not None:
            return task.write_deadline
        return task.progress_deadline

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._drain_pending()
            now_m = time.monotonic()
            timeout: Optional[float] = None
            if self._active:
                earliest = min(self._task_deadline(t) for t in self._active)
                timeout = max(0.0, earliest - now_m)
            try:
                events = self._selector.select(timeout=timeout)
            except OSError:
                break
            now_m = time.monotonic()
            for key, mask in events:
                tag = key.data
                if tag is None:
                    try:
                        while self._wake_r.recv(4096):
                            pass
                    except BlockingIOError:
                        pass
                    except OSError:
                        pass
                    continue
                kind, task = tag
                if task not in self._active:
                    continue
                try:
                    if kind == "up":
                        self._on_upstream_ready(task)
                    else:
                        if mask & selectors.EVENT_WRITE:
                            self._on_client_writable(task)
                        if task in self._active and (mask & selectors.EVENT_READ):
                            self._on_client_readable(task)
                except Exception:
                    task.upstream_read_error = True
                    self._finish_task(task)
            if self._active:
                expired = [
                    t for t in self._active if now_m >= self._task_deadline(t)
                ]
                for task in expired:
                    if task.write_buf and task.write_deadline is not None:
                        task.downstream_write_error = True
                    else:
                        task.upstream_read_error = True
                    self._finish_task(task)

        for task in list(self._active):
            task.downstream_write_error = True
            self._finish_task(task)
        with self._pending_lock:
            while self._pending:
                task = self._pending.popleft()
                task.downstream_write_error = True
                self._finish_task(task)


class FreeTPURouterServer(ThreadingHTTPServer):
    """Multi-threaded stdlib HTTP server hosting `RouterEngine`."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024

    def __init__(
        self,
        server_address: Tuple[str, int],
        engine: RouterEngine,
        *,
        verbose: bool = False,
    ) -> None:
        self.engine = engine
        self.verbose = verbose
        clean_host = str(server_address[0]).strip().strip("[]")
        self.configured_bind_host = clean_host.lower()
        try:
            ip_obj = ipaddress.ip_address(clean_host.split("%", 1)[0])
        except ValueError:
            ip_obj = None
        if ip_obj is not None and ip_obj.version == 6:
            self.address_family = socket.AF_INET6
        else:
            self.address_family = socket.AF_INET
        self._conn_semaphore = threading.BoundedSemaphore(
            engine.config.max_concurrent_connections
        )
        self.sse_reactor = _SSEReactor()
        super().__init__((clean_host, server_address[1]), _RouterRequestHandler)

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
        self.sse_reactor.start()
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.sse_reactor.stop()
            self.engine.stop_background_poller()

    def server_close(self) -> None:
        self.sse_reactor.stop()
        self.engine.stop_background_poller()
        super().server_close()


class _RouterRequestHandler(BaseHTTPRequestHandler):
    server: FreeTPURouterServer
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def setup(self) -> None:
        self.timeout = self.server.engine.config.client_header_timeout_s
        super().setup()
        header_deadline = (
            time.monotonic() + self.server.engine.config.client_header_timeout_s
        )
        self.rfile = _DeadlineReader(
            self.rfile,
            self.connection,
            header_deadline,
            max_header_bytes=self.server.engine.config.max_header_bytes,
        )

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except _HeaderSizeLimitExceeded:
            self.close_connection = True
            if not hasattr(self, "requestline"):
                self.requestline = ""
            if not hasattr(self, "headers"):
                self.headers = http.client.HTTPMessage()
            self._safe_send_bytes(
                431,
                "application/json",
                _openai_error_body(
                    "Request headers exceed maximum size of "
                    f"{self.server.engine.config.max_header_bytes} bytes",
                    error_type="invalid_request_error",
                    code="request_header_fields_too_large",
                ),
            )
            try:
                self.connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            message = (format % args).translate(_LOG_CONTROL_CHAR_TABLE)
            sys.stderr.write(
                f"{self.address_string()} - - [{self.log_date_time_string()}] {message}\n"
            )

    def _trusted_listener_hosts(self) -> set[str]:
        """Return hostnames/IPs trusted as direct same-origin listener authorities."""
        trusted: set[str] = set()
        configured_host = str(
            getattr(self.server, "configured_bind_host", "")
        ).strip().lower().strip("[]")
        if configured_host and configured_host not in ("0.0.0.0", "::"):
            trusted.add(configured_host)
            if is_loopback_host(configured_host):
                trusted.update({"localhost", "127.0.0.1", "::1"})
        bound_host = str(self.server.server_address[0]).strip().lower().strip("[]")
        if is_loopback_host(bound_host):
            trusted.update({"localhost", "127.0.0.1", "::1", bound_host})
        elif bound_host in ("0.0.0.0", "::", ""):
            trusted.update({"localhost", "127.0.0.1", "::1"})
            sock = getattr(self, "connection", None)
            if sock is not None:
                try:
                    local_ip = str(sock.getsockname()[0]).strip().lower().strip("[]")
                    if local_ip:
                        trusted.add(local_ip)
                except OSError:
                    pass
        else:
            trusted.add(bound_host)
            sock = getattr(self, "connection", None)
            if sock is not None:
                try:
                    local_ip = str(sock.getsockname()[0]).strip().lower().strip("[]")
                    if local_ip:
                        trusted.add(local_ip)
                except OSError:
                    pass
        return trusted

    def _validate_request_host(self) -> bool:
        """Validate Host header against trusted listener authorities or configured public_origin."""
        headers = getattr(self, "headers", None)
        if headers is None:
            return False
        raw_host = headers.get("Host")
        if not raw_host:
            return False
        host_val = raw_host.strip()
        if (
            not host_val
            or any(ch.isspace() for ch in host_val)
            or "/" in host_val
            or "@" in host_val
        ):
            return False
        cfg = self.server.engine.config
        if cfg.public_origin:
            pub_norm = normalize_origin(cfg.public_origin)
            if pub_norm is not None:
                candidate_pub = normalize_origin(f"{pub_norm[0]}://{host_val}")
                if candidate_pub is not None and candidate_pub == pub_norm:
                    return True
        eff_scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
        if cfg.trust_proxy_headers:
            xfp = headers.get("X-Forwarded-Proto")
            if xfp:
                candidate = xfp.split(",", 1)[0].strip().lower()
                if candidate in VALID_URL_SCHEMES:
                    eff_scheme = candidate
        host_origin = normalize_origin(f"{eff_scheme}://{host_val}")
        if host_origin is None:
            return False
        _, host_name, _ = host_origin
        if host_name not in self._trusted_listener_hosts():
            return False
        parts = urlsplit(f"{eff_scheme}://{host_val}")
        bound_port = int(self.server.server_address[1])
        return parts.port is None or parts.port == bound_port

    def _reject_invalid_host_if_needed(self) -> bool:
        """Return True and send HTTP 403 if the request Host header is untrusted."""
        if self._validate_request_host():
            return False
        self._send_bytes(
            403,
            "application/json",
            _openai_error_body(
                "Request Host header is not allowed by router listener policy",
                error_type="permission_error",
                code="invalid_host",
            ),
        )
        return True

    def _is_same_origin(self, origin: str) -> bool:
        """Return True if `origin` matches configured `public_origin` or a trusted listener authority."""
        parsed_origin = normalize_origin(origin)
        if parsed_origin is None:
            return False
        cfg = self.server.engine.config
        if cfg.public_origin:
            pub_norm = normalize_origin(cfg.public_origin)
            if pub_norm is not None and parsed_origin == pub_norm:
                return True
        headers = getattr(self, "headers", None)
        if headers is None:
            return False
        raw_host = headers.get("Host")
        if not raw_host:
            return False
        host_val = raw_host.strip()
        if not host_val or any(ch.isspace() for ch in host_val) or "/" in host_val or "@" in host_val:
            return False
        eff_scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
        if cfg.trust_proxy_headers:
            xfp = headers.get("X-Forwarded-Proto")
            if xfp:
                candidate = xfp.split(",", 1)[0].strip().lower()
                if candidate in VALID_URL_SCHEMES:
                    eff_scheme = candidate
        host_origin = normalize_origin(f"{eff_scheme}://{host_val}")
        if host_origin is None or parsed_origin != host_origin:
            return False
        _, host_name, host_port = host_origin
        bound_port = int(self.server.server_address[1])
        if host_name not in self._trusted_listener_hosts():
            return False
        return host_port == bound_port

    def _is_client_disconnected(self) -> bool:
        """Non-blocking probe detecting whether the downstream TCP connection aborted.

        Per RFC 9112 §9.6, an orderly read-side EOF (`shutdown(SHUT_WR)`) is not a
        disconnect once the request body has been read; only socket errors/resets
        or invalid descriptors abort retry/failover.
        """
        sock = getattr(self, "connection", None)
        if sock is None:
            return True
        try:
            readable, errored = wait_sockets_readable([sock], 0.0)
        except ValueError:
            try:
                return sock.fileno() < 0
            except (OSError, AttributeError):
                return True
        except OSError:
            return True
        if sock in errored:
            return True
        if sock not in readable:
            return False
        try:
            sock.recv(1, socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0))
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            return True
        return False

    def _resolve_cors_header(self) -> Tuple[bool, Optional[str]]:
        """Check request Origin against explicit `cors_allowed_origins` and same-origin authority.

        Returns `(allowed, acao_header_value)`. Non-browser requests without an
        `Origin` header and same-origin requests are allowed with
        `acao_header_value=None` unless also explicitly allowlisted for CORS.
        """
        headers = getattr(self, "headers", None)
        if headers is None:
            return True, None
        raw_origin = headers.get("Origin")
        if raw_origin is None:
            return True, None
        origin = raw_origin.strip()
        allowed = self.server.engine.config.cors_allowed_origins
        if allowed:
            if "*" in allowed:
                return True, "*"
            if origin.rstrip("/") in allowed:
                return True, origin
        if self._is_same_origin(origin):
            return True, None
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

    def _read_request_body(
        self,
        content_len: int,
        *,
        reserved_out: Optional[List[int]] = None,
    ) -> Tuple[Optional[bytes], Optional[Tuple[int, str, str]]]:
        """Read exactly `content_len` bytes within the configured wall-clock deadline.

        Charges upload memory incrementally as bytes actually arrive (plus the
        temporary `bytes(buf)` copy during finalization) so stalled uploads do
        not pin the router's shared request memory budget.
        """
        engine = self.server.engine
        timeout_s = engine.config.client_read_timeout_s
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        remaining = content_len
        reserved_local = 0
        read_fn = getattr(self.rfile, "read1", self.rfile.read)
        try:
            while remaining > 0:
                time_left = deadline - time.monotonic()
                if time_left <= 0:
                    buf.clear()
                    return None, (408, "request_timeout", "Request body read timed out")
                self.connection.settimeout(time_left)
                chunk = read_fn(min(65536, remaining))
                if not chunk:
                    buf.clear()
                    return None, (400, "incomplete_body", "Incomplete request body")
                chunk_len = len(chunk)
                if not engine.try_reserve_request_memory(chunk_len):
                    buf.clear()
                    return None, (
                        503,
                        "server_memory_limit_reached",
                        "Router request memory budget exhausted; retry later",
                    )
                reserved_local += chunk_len
                if reserved_out is not None:
                    reserved_out[0] += chunk_len
                buf.extend(chunk)
                remaining -= chunk_len
        except TimeoutError:
            buf.clear()
            return None, (408, "request_timeout", "Request body read timed out")
        except OSError:
            buf.clear()
            return None, (400, "incomplete_body", "Client disconnected while uploading body")
        copy_extra = max(0, (2 * len(buf)) - reserved_local)
        if copy_extra > 0:
            if not engine.try_reserve_request_memory(copy_extra):
                buf.clear()
                return None, (
                    503,
                    "server_memory_limit_reached",
                    "Router request memory budget exhausted; retry later",
                )
            reserved_local += copy_extra
            if reserved_out is not None:
                reserved_out[0] += copy_extra
        try:
            return bytes(buf), None
        finally:
            buf.clear()
            if copy_extra > 0:
                engine.release_request_memory(copy_extra)
                if reserved_out is not None:
                    reserved_out[0] = max(0, reserved_out[0] - copy_extra)

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
        if self._reject_invalid_host_if_needed():
            return
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
        self.send_header("Access-Control-Allow-Headers", _CORS_ALLOWED_HEADERS)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self._reject_invalid_host_if_needed():
            return
        path = urlsplit(self.path).path
        ctx = self._authorize(path, "GET")
        if ctx is None:
            return

        if path == "/health":
            detailed = ctx.metadata.get("authenticated", True) is not False
            code, payload = self.server.engine.health_payload(detailed=detailed)
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
        if self._reject_invalid_host_if_needed():
            return
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

        engine = self.server.engine
        max_total_req_mem = engine.config.max_total_request_memory_bytes
        min_parse_budget = max(1, content_len * REQUEST_MEMORY_PARSE_MULTIPLIER)
        if min_parse_budget > max_total_req_mem:
            self._send_bytes(
                413,
                "application/json",
                _openai_error_body(
                    f"Request estimated parse memory ({min_parse_budget} bytes) exceeds router request memory budget ({max_total_req_mem} bytes)",
                    code="payload_too_large",
                ),
            )
            return

        req_mem_reserved = [0]
        try:
            raw_req, read_err = self._read_request_body(
                content_len,
                reserved_out=req_mem_reserved,
            )
            if read_err is not None or raw_req is None:
                err_status, err_code, err_msg = read_err or (
                    400,
                    "incomplete_body",
                    "Failed to read request body",
                )
                err_type = (
                    "service_unavailable"
                    if err_status == 503
                    else "invalid_request_error"
                )
                self._safe_send_bytes(
                    err_status,
                    "application/json",
                    _openai_error_body(err_msg, error_type=err_type, code=err_code),
                )
                return

            # Allow legitimate long-running inference/streaming responses to write cleanly.
            try:
                write_timeout = max(
                    engine.config.client_read_timeout_s,
                    engine.config.request_timeout_s,
                )
                self.connection.settimeout(write_timeout)
            except OSError:
                pass

            req_containers = [0]
            try:
                req_nodes = _validate_json_structure(
                    raw_req,
                    max_depth=engine.config.max_json_depth,
                    max_nodes=engine.config.max_json_nodes,
                    containers_out=req_containers,
                )
            except _JsonStructureLimitError as exc:
                self._send_bytes(
                    400,
                    "application/json",
                    _openai_error_body(
                        str(exc),
                        code="json_complexity_limit_exceeded",
                    ),
                )
                return

            needed_parse_mem = _estimate_json_parse_memory(
                raw_req,
                nodes=req_nodes,
                containers=req_containers[0],
            )
            if needed_parse_mem > max_total_req_mem:
                self._send_bytes(
                    413,
                    "application/json",
                    _openai_error_body(
                        f"Request estimated parse memory ({needed_parse_mem} bytes) exceeds router request memory budget ({max_total_req_mem} bytes)",
                        code="payload_too_large",
                    ),
                )
                return
            if needed_parse_mem > req_mem_reserved[0]:
                extra_parse_mem = needed_parse_mem - req_mem_reserved[0]
                if not engine.try_reserve_request_memory(extra_parse_mem):
                    self._send_bytes(
                        503,
                        "application/json",
                        _openai_error_body(
                            "Router request memory budget exhausted; retry later",
                            error_type="service_unavailable",
                            code="server_memory_limit_reached",
                        ),
                    )
                    return
                req_mem_reserved[0] += extra_parse_mem

            parsed_fields, parse_err = _parse_inference_request_fields(raw_req, path)
            if parse_err is not None or parsed_fields is None:
                err_status, err_code, err_msg = parse_err or (
                    400,
                    None,
                    "Malformed JSON request",
                )
                self._send_bytes(
                    err_status,
                    "application/json",
                    _openai_error_body(err_msg, code=err_code),
                )
                return
            (
                model,
                stream,
                est_prompt_tokens,
                expected_choices,
                expect_final_usage,
            ) = parsed_fields
            # All decoded JSON containers went out of scope inside
            # `_parse_inference_request_fields`; release the parse reservation now.
            parse_overhead = req_mem_reserved[0] - content_len
            if parse_overhead > 0:
                engine.release_request_memory(parse_overhead)
                req_mem_reserved[0] = content_len

            self._handle_inference(
                path,
                raw_req,
                model,
                stream,
                est_prompt_tokens,
                expected_choices,
                ctx,
                expect_final_usage=expect_final_usage,
            )
        finally:
            if req_mem_reserved[0] > 0:
                engine.release_request_memory(req_mem_reserved[0])
                req_mem_reserved[0] = 0

    def _handle_inference(
        self,
        endpoint: str,
        raw_req: bytes,
        model: str,
        stream: bool,
        est_prompt_tokens: int,
        expected_choices: int,
        auth_ctx: AuthContext,
        *,
        expect_final_usage: bool = False,
    ) -> None:
        engine = self.server.engine
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        req_start = time.monotonic()
        policy_override = self.headers.get("X-FreeTPU-Policy")
        if policy_override:
            policy_override = policy_override.strip()

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
            if tried_ids and self._is_client_disconnected():
                # Caller already cancelled; do not start billable fallback generation.
                self.close_connection = True
                if last_attempted_backend is not None:
                    engine._record_terminal_failure(
                        request_id=req_id,
                        model=model,
                        backend_id=last_attempted_backend.config.id,
                        matched_backend=True,
                        status_code=499,
                        latency_ms=(time.monotonic() - req_start) * 1000.0,
                        stream=stream,
                        failovers=failovers,
                        principal=auth_ctx.principal,
                    )
                return
            candidates = engine.select_candidates(
                model,
                policy=policy_override,
                exclude_ids=tried_ids,
            )
            backend: Optional[BackendState] = None
            slot_lease: Optional[_BackendSlotLease] = None
            total_inflight_now: Optional[int] = None
            attempt_start = time.monotonic()
            for cand in candidates:
                if cand.config.id in tried_ids:
                    continue
                attempt_start = time.monotonic()
                lease_res = engine.reserve_backend_lease(
                    cand, now_monotonic=attempt_start
                )
                if lease_res is not None:
                    slot_lease, total_inflight_now = lease_res
                    backend = cand
                    break
            if backend is None or slot_lease is None or total_inflight_now is None:
                break

            non_stream_result: Optional[
                Tuple[int, str, bytes, float, float, int, int]
            ] = None
            conn: Optional[http.client.HTTPConnection] = None
            resp: Optional[http.client.HTTPResponse] = None
            raw_sock: Any = None
            deadline_sock: Optional[DeadlineSocket] = None
            committed_flag = [False]
            resp_mem_reserved = [0]
            try:
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
                scheme, host, port, upstream_path = _join_backend_url(
                    backend.config.base_url, endpoint
                )
                timeout_s = backend.config.timeout_s or engine.config.request_timeout_s
                attempt_deadline = attempt_start + timeout_s
                conn = _open_http_connection(scheme, host, port, timeout_s)
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream" if stream else "application/json",
                    "Connection": "close",
                    "X-FreeTPU-Request-Id": req_id,
                }
                api_key = backend.config.resolved_api_key()
                _ensure_safe_credential_transport(backend.config, scheme, host, api_key)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                conn.request("POST", upstream_path, body=raw_req, headers=headers)
                raw_sock = conn.sock
                if raw_sock is not None:
                    try:
                        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except OSError:
                        pass
                rem_hdr = attempt_deadline - time.monotonic()
                if rem_hdr <= 0:
                    raise TimeoutError("Upstream response headers exceeded elapsed deadline")
                if raw_sock is not None:
                    deadline_sock = DeadlineSocket(
                        raw_sock,
                        attempt_deadline,
                        label=f"Backend {backend.config.id!r} response",
                        client_sock=self.connection,
                    )
                    conn.sock = deadline_sock
                resp = conn.getresponse()
                upstream_sock = deadline_sock if deadline_sock is not None else raw_sock

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
                    sse_outcome, sse_err_status, sse_err_body = self._proxy_sse_stream(
                        resp=resp,
                        upstream_sock=deadline_sock if deadline_sock is not None else upstream_sock,
                        timeout_s=timeout_s,
                        backend=backend,
                        model=model,
                        expected_choices=expected_choices,
                        req_id=req_id,
                        req_start=req_start,
                        attempt_start=attempt_start,
                        failovers=failovers,
                        est_prompt_tokens=est_prompt_tokens,
                        principal=auth_ctx.principal,
                        expect_final_usage=expect_final_usage,
                        committed_flag=committed_flag,
                    )
                    if sse_outcome == "committed":
                        return
                    if sse_outcome == "memory_limit_exceeded":
                        # Shared response memory budget exhausted: request-scoped 503 without backend cooldown.
                        latency_ms = (time.monotonic() - req_start) * 1000.0
                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                        non_stream_result = (
                            503,
                            "application/json",
                            sse_err_body
                            or _openai_error_body(
                                "Router response memory budget exhausted; retry later",
                                error_type="service_unavailable",
                                code="server_memory_limit_reached",
                            ),
                            latency_ms,
                            attempt_ms,
                            0,
                            0,
                        )
                    elif sse_outcome == "response_too_large":
                        # Pre-commit SSE size limit is request-scoped: do not penalize backend health or fail over.
                        latency_ms = (time.monotonic() - req_start) * 1000.0
                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                        non_stream_result = (
                            502,
                            "application/json",
                            sse_err_body
                            or _openai_error_body(
                                "Upstream SSE response exceeded maximum size limit",
                                error_type="bad_gateway",
                                code="response_too_large",
                            ),
                            latency_ms,
                            attempt_ms,
                            0,
                            0,
                        )
                    elif sse_outcome == "client_error":
                        latency_ms = (time.monotonic() - req_start) * 1000.0
                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                        non_stream_result = (
                            sse_err_status,
                            "application/json",
                            sse_err_body
                            or _openai_error_body(
                                "Upstream rejected request",
                                error_type="invalid_request_error",
                            ),
                            latency_ms,
                            attempt_ms,
                            0,
                            0,
                        )
                    else:
                        # Pre-first-token stream failure or retryable SSE error: fail over.
                        if sse_outcome == "retryable_error":
                            reason = "upstream_sse_error"
                            last_error_status = sse_err_status
                            last_error_msg = (
                                f"Backend {backend.config.id!r} returned SSE error ({sse_err_status})"
                            )
                            last_error_type = "server_error"
                            last_error_code = "upstream_sse_error"
                            last_error_body = sse_err_body
                            last_error_content_type = "application/json"
                        else:
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
                else:
                    # Non-streaming response (or non-retryable error status such as 4xx)
                    max_resp_bytes = (
                        engine.config.max_response_body_bytes
                        if 200 <= resp.status < 300
                        else min(
                            engine.config.max_response_body_bytes,
                            MAX_ERROR_RESPONSE_BYTES,
                        )
                    )
                    try:
                        raw_resp = _read_bounded_http_response(
                            resp,
                            upstream_sock,
                            max_bytes=max_resp_bytes,
                            deadline=attempt_deadline,
                            label=f"Backend {backend.config.id!r} response",
                            reserve_fn=engine.try_reserve_response_memory,
                            release_fn=engine.release_response_memory,
                            reserved_out=resp_mem_reserved,
                        )
                    except _MemoryBudgetExceededError:
                        # Shared response memory budget exhausted: request-scoped 503 without backend cooldown.
                        if resp_mem_reserved[0] > 0:
                            engine.release_response_memory(resp_mem_reserved[0])
                            resp_mem_reserved[0] = 0
                        latency_ms = (time.monotonic() - req_start) * 1000.0
                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                        non_stream_result = (
                            503,
                            "application/json",
                            _openai_error_body(
                                "Router response memory budget exhausted; retry later",
                                error_type="service_unavailable",
                                code="server_memory_limit_reached",
                            ),
                            latency_ms,
                            attempt_ms,
                            0,
                            0,
                        )
                    except _UpstreamResponseTooLargeError:
                        # Response-size cap is request-scoped: do not penalize backend health or fail over.
                        if resp_mem_reserved[0] > 0:
                            engine.release_response_memory(resp_mem_reserved[0])
                            resp_mem_reserved[0] = 0
                        latency_ms = (time.monotonic() - req_start) * 1000.0
                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                        non_stream_result = (
                            502,
                            "application/json",
                            _openai_error_body(
                                f"Upstream response exceeded maximum size of {max_resp_bytes} bytes",
                                error_type="bad_gateway",
                                code="response_too_large",
                            ),
                            latency_ms,
                            attempt_ms,
                            0,
                            0,
                        )
                    else:
                        prompt_tokens_val = 0
                        completion_tokens_val = 0
                        if 200 <= resp.status < 300:
                            # Validate JSON structure on raw bytes first, reserve exact
                            # PEP-393 peak parse memory before decoding, extract usage,
                            # and drop the parsed tree immediately.
                            resp_nodes: Optional[int] = None
                            resp_containers = [0]
                            try:
                                resp_nodes = _validate_json_structure(
                                    raw_resp,
                                    max_depth=engine.config.max_json_depth,
                                    max_nodes=engine.config.max_response_json_nodes,
                                    containers_out=resp_containers,
                                )
                            except _JsonStructureLimitError:
                                if resp_mem_reserved[0] > 0:
                                    engine.release_response_memory(resp_mem_reserved[0])
                                    resp_mem_reserved[0] = 0
                                latency_ms = (time.monotonic() - req_start) * 1000.0
                                attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                                non_stream_result = (
                                    502,
                                    "application/json",
                                    _openai_error_body(
                                        "Upstream response JSON exceeded structural complexity limits",
                                        error_type="bad_gateway",
                                        code="response_too_large",
                                    ),
                                    latency_ms,
                                    attempt_ms,
                                    0,
                                    0,
                                )
                            if resp_nodes is not None:
                                needed_resp_mem = _estimate_json_parse_memory(
                                    raw_resp,
                                    nodes=resp_nodes,
                                    containers=resp_containers[0],
                                )
                                parse_extra = max(0, needed_resp_mem - resp_mem_reserved[0])
                                if parse_extra > 0:
                                    if not engine.try_reserve_response_memory(parse_extra):
                                        if resp_mem_reserved[0] > 0:
                                            engine.release_response_memory(resp_mem_reserved[0])
                                            resp_mem_reserved[0] = 0
                                        latency_ms = (time.monotonic() - req_start) * 1000.0
                                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                                        non_stream_result = (
                                            503,
                                            "application/json",
                                            _openai_error_body(
                                                "Router response memory budget exhausted; retry later",
                                                error_type="service_unavailable",
                                                code="server_memory_limit_reached",
                                            ),
                                            latency_ms,
                                            attempt_ms,
                                            0,
                                            0,
                                        )
                                        parse_extra = -1
                                    else:
                                        resp_mem_reserved[0] += parse_extra
                                if parse_extra >= 0:
                                    try:
                                        resp_text = raw_resp.decode("utf-8")
                                        loaded_resp = json.loads(resp_text)
                                        del resp_text
                                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                        raise http.client.IncompleteRead(raw_resp) from exc
                                    else:
                                        if not isinstance(loaded_resp, Mapping):
                                            raise http.client.IncompleteRead(raw_resp)
                                        prompt_tokens_val, completion_tokens_val = (
                                            _extract_non_stream_usage(
                                                raw_resp,
                                                est_prompt_tokens,
                                                parsed_resp=loaded_resp,
                                            )
                                        )
                                        del loaded_resp
                                        if parse_extra > 0:
                                            engine.release_response_memory(parse_extra)
                                            resp_mem_reserved[0] = max(
                                                0, resp_mem_reserved[0] - parse_extra
                                            )
                                        latency_ms = (time.monotonic() - req_start) * 1000.0
                                        attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                                        content_type = (
                                            resp.getheader("Content-Type") or "application/json"
                                        )
                                        non_stream_result = (
                                            resp.status,
                                            content_type,
                                            raw_resp,
                                            latency_ms,
                                            attempt_ms,
                                            prompt_tokens_val,
                                            completion_tokens_val,
                                        )
                        else:
                            latency_ms = (time.monotonic() - req_start) * 1000.0
                            attempt_ms = (time.monotonic() - attempt_start) * 1000.0
                            content_type = resp.getheader("Content-Type") or "application/json"
                            non_stream_result = (
                                resp.status,
                                content_type,
                                raw_resp,
                                latency_ms,
                                attempt_ms,
                                0,
                                0,
                            )
            except ClientDisconnectedError:
                # Downstream client disconnected mid-wait: cancel upstream immediately
                # via finally without penalizing backend health or triggering failover.
                if resp_mem_reserved[0] > 0:
                    engine.release_response_memory(resp_mem_reserved[0])
                    resp_mem_reserved[0] = 0
                self.close_connection = True
                if not committed_flag[0]:
                    engine._record_terminal_failure(
                        request_id=req_id,
                        model=model,
                        backend_id=backend.config.id,
                        matched_backend=True,
                        status_code=499,
                        latency_ms=(time.monotonic() - req_start) * 1000.0,
                        stream=stream,
                        failovers=failovers,
                        principal=auth_ctx.principal,
                    )
                return
            except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
                if resp_mem_reserved[0] > 0:
                    engine.release_response_memory(resp_mem_reserved[0])
                    resp_mem_reserved[0] = 0
                if committed_flag[0]:
                    # Downstream HTTP headers were already committed; never retry, fail over,
                    # or write a second HTTP response into the open stream.
                    self.close_connection = True
                    return
                if isinstance(exc, TimeoutError):
                    reason = "timeout"
                    last_error_status = 504
                else:
                    reason = "connection_error"
                    last_error_status = 502
                safe_detail = _safe_probe_error(exc)
                last_error_msg = f"Backend {backend.config.id!r} failed ({safe_detail})"
                last_error_type = "service_unavailable"
                last_error_code = reason
                last_error_body = None
                last_error_content_type = "application/json"
                engine.mark_backend_failed(backend, reason)
                prev_failure = (backend.config.id, reason)
                continue
            finally:
                if non_stream_result is None and resp_mem_reserved[0] > 0:
                    engine.release_response_memory(resp_mem_reserved[0])
                    resp_mem_reserved[0] = 0
                slot_lease.release()
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
                if raw_sock is not None:
                    try:
                        raw_sock.close()
                    except OSError:
                        pass

            if non_stream_result is not None:
                try:
                    (
                        resp_status,
                        content_type,
                        raw_resp,
                        latency_ms,
                        attempt_ms,
                        prompt_tokens,
                        completion_tokens,
                    ) = non_stream_result
                    if 200 <= resp_status < 300:
                        with engine._lock:
                            backend.record_latency(attempt_ms)

                    engine._record_usage(
                        request_id=req_id,
                        model=model,
                        backend=backend,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        ttft_ms=latency_ms if 200 <= resp_status < 300 else None,
                        stream=stream,
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
                finally:
                    if resp_mem_reserved[0] > 0:
                        engine.release_response_memory(resp_mem_reserved[0])
                        resp_mem_reserved[0] = 0
                return

        total_latency_ms = (time.monotonic() - req_start) * 1000.0
        if last_attempted_backend is None:
            engine._emit_queue_pressure(
                model=model,
                active_inflight=engine.metrics.total_inflight(),
                healthy_backends=0,
                reason="no_healthy_backend",
            )
            engine._record_terminal_failure(
                request_id=req_id,
                model=model,
                backend_id="none",
                matched_backend=False,
                status_code=503,
                latency_ms=total_latency_ms,
                stream=stream,
                failovers=failovers,
                principal=auth_ctx.principal,
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

        engine._record_terminal_failure(
            request_id=req_id,
            model=model,
            backend_id=last_attempted_backend.config.id,
            matched_backend=True,
            status_code=last_error_status,
            latency_ms=total_latency_ms,
            stream=stream,
            failovers=failovers,
            principal=auth_ctx.principal,
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
        expect_final_usage: bool = False,
        committed_flag: Optional[List[bool]] = None,
    ) -> Tuple[str, int, Optional[bytes]]:
        """Buffer until first generated output or terminator before committing 200 OK.

        Uses incremental bounded reads via `_BoundedSSEReader` during pre-output
        buffering (governed by byte and time limits, without an arbitrary event
        count cap so prompt-progress streams succeed), classifies pre-commit SSE
        error events so retryable backend errors fail over and client errors pass
        through without marking success, and enforces a post-commit progress
        deadline that ignores keepalive comments.

        Returns `(outcome, status_code, error_body)` where `outcome` is one of
        `"committed"`, `"retryable_error"`, `"client_error"`, or `"aborted"`.
        """
        engine = self.server.engine
        sse_mem_box = [0]
        sse_reader = _BoundedSSEReader(
            resp,
            upstream_sock,
            reserve_fn=engine.try_reserve_response_memory,
            available_fn=engine.available_response_memory,
            reserved_out=sse_mem_box,
        )
        pre_commit_buf = bytearray()
        pre_commit_bytes = 0
        max_pre_commit_bytes = engine.config.max_sse_pre_commit_bytes
        first_chunk_time: Optional[float] = None
        pending_event_start = 0
        pending_event_has_data = False
        pending_event_type = ""
        pre_commit_skip_leading_lf = False
        pre_commit_deadline = attempt_start + timeout_s

        def _extract_pending_event_payload(end_pos: int) -> bytes:
            data_parts: List[bytes] = []
            scan = pending_event_start
            while scan < end_pos:
                nl = pre_commit_buf.find(b"\n", scan, end_pos)
                cr = pre_commit_buf.find(b"\r", scan, end_pos)
                if nl == -1 and cr == -1:
                    ln_end = end_pos
                elif cr != -1 and (nl == -1 or cr < nl):
                    ln_end = cr + 2 if (cr + 1 < end_pos and pre_commit_buf[cr + 1] == 10) else cr + 1
                else:
                    ln_end = nl + 1
                cline = bytes(pre_commit_buf[scan:ln_end]).rstrip(b"\r\n")
                if cline.startswith(b"data:"):
                    dval = cline[5:]
                    if dval.startswith(b" "):
                        dval = dval[1:]
                    data_parts.append(dval)
                scan = ln_end
            if not data_parts:
                return b""
            if len(data_parts) == 1:
                return data_parts[0].strip()
            return b"\n".join(data_parts).strip()

        def release_sse_mem(amount: int) -> None:
            if amount <= 0:
                return
            rel = min(amount, sse_mem_box[0])
            if rel > 0:
                engine.release_response_memory(rel)
                sse_mem_box[0] -= rel

        try:
            # Phase 1: upstream-only incremental read through HTTPResponse.read1().
            # Enforce byte cap, shared memory budget, and elapsed deadline as bytes arrive.
            while True:
                remaining_bytes = max_pre_commit_bytes - pre_commit_bytes
                if remaining_bytes <= 0:
                    return (
                        "response_too_large",
                        502,
                        _openai_error_body(
                            f"Upstream SSE event exceeded maximum size of {max_pre_commit_bytes} bytes",
                            error_type="bad_gateway",
                            code="response_too_large",
                        ),
                    )
                try:
                    line, exceeded_cap = sse_reader.readline(
                        deadline=pre_commit_deadline,
                        max_bytes=remaining_bytes,
                    )
                except _MemoryBudgetExceededError:
                    return (
                        "memory_limit_exceeded",
                        503,
                        _openai_error_body(
                            "Router response memory budget exhausted; retry later",
                            error_type="service_unavailable",
                            code="server_memory_limit_reached",
                        ),
                    )
                if exceeded_cap:
                    return (
                        "response_too_large",
                        502,
                        _openai_error_body(
                            f"Upstream SSE event exceeded maximum size of {max_pre_commit_bytes} bytes",
                            error_type="bad_gateway",
                            code="response_too_large",
                        ),
                    )
                if not line:
                    # EOF before generated output/terminator -> retryable failure.
                    return "aborted", 502, None
                pre_commit_bytes += len(line)
                pre_commit_buf.extend(line)
                if pre_commit_skip_leading_lf and line == b"\n":
                    pre_commit_skip_leading_lf = False
                    if not pending_event_has_data:
                        pending_event_start = len(pre_commit_buf)
                    continue
                pre_commit_skip_leading_lf = line.endswith(b"\r")
                if line in (b"\n", b"\r\n", b"\r"):
                    if pending_event_has_data:
                        payload_bytes = _extract_pending_event_payload(
                            len(pre_commit_buf) - len(line)
                        )
                        ev_type = pending_event_type
                        pending_event_has_data = False
                        pending_event_type = ""
                        pending_event_start = len(pre_commit_buf)
                        err_info = _classify_sse_error_event(ev_type, payload_bytes)
                        if err_info is not None:
                            is_retryable, err_status, err_body = err_info
                            if is_retryable:
                                return "retryable_error", err_status, err_body
                            return "client_error", err_status, err_body
                        try:
                            should_commit = _sse_event_should_commit(payload_bytes)
                        except _MalformedSSEPayloadError:
                            return (
                                "retryable_error",
                                502,
                                _openai_error_body(
                                    "Upstream SSE backend emitted malformed JSON payload before first token",
                                    error_type="bad_gateway",
                                    code="upstream_stream_error",
                                ),
                            )
                        if should_commit:
                            first_chunk_time = time.monotonic()
                            break
                    else:
                        pending_event_type = ""
                        pending_event_start = len(pre_commit_buf)
                    continue
                content_line = line.rstrip(b"\r\n")
                if content_line.startswith(b":"):
                    continue
                if content_line.startswith(b"event:"):
                    pending_event_type = content_line[6:].strip().decode("utf-8", errors="replace")
                elif content_line.startswith(b"data:"):
                    pending_event_has_data = True

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
            saw_final_usage = False
            saw_timings_completion = False
            saw_done_sentinel = False
            saw_upstream_error_event = False
            finished_choice_indices: set[int] = set()
            upstream_read_error = False
            size_limit_exceeded = False
            memory_limit_exceeded = False
            downstream_write_error = False
            event_data_lines: List[bytes] = []
            current_event_type = ""
            required_choices = max(1, expected_choices)

            def process_sse_event_from(
                ev_type: str,
                data_lines: List[bytes],
            ) -> Tuple[bool, bool]:
                """Process one complete blank-line-terminated SSE event.

                Returns `(made_progress, saw_done)`.
                """
                nonlocal first_content_ttft_ms, prompt_tokens, completion_tokens
                nonlocal est_completion_chars, saw_explicit_prompt_usage
                nonlocal saw_explicit_completion_usage, saw_final_usage, saw_timings_completion
                nonlocal saw_done_sentinel, saw_upstream_error_event
                if not data_lines:
                    return False, False
                payload_bytes = (
                    data_lines[0].strip()
                    if len(data_lines) == 1
                    else b"\n".join(data_lines).strip()
                )
                if not payload_bytes:
                    return False, False
                if payload_bytes == b"[DONE]":
                    saw_done_sentinel = True
                    return True, True
                try:
                    obj = json.loads(payload_bytes)
                except UnicodeDecodeError:
                    try:
                        obj = json.loads(payload_bytes.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        if ev_type.strip().lower() == "error":
                            saw_upstream_error_event = True
                            return False, False
                        return True, False
                except json.JSONDecodeError:
                    if ev_type.strip().lower() == "error":
                        saw_upstream_error_event = True
                        return False, False
                    return True, False
                if not isinstance(obj, dict):
                    if ev_type.strip().lower() == "error":
                        saw_upstream_error_event = True
                        return False, False
                    return True, False
                if (
                    ev_type.strip().lower() == "error"
                    or obj.get("error") is not None
                    or obj.get("object") == "error"
                ):
                    saw_upstream_error_event = True
                    return False, False
                made_progress = False
                choices = obj.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                            made_progress = True
                            c_idx = choice.get("index")
                            if isinstance(c_idx, int) and not isinstance(c_idx, bool) and c_idx >= 0:
                                finished_choice_indices.add(c_idx)
                            else:
                                finished_choice_indices.add(len(finished_choice_indices))
                output_txt = _extract_choice_output_text(obj)
                has_audio_out = _has_choice_audio_output(obj)
                if output_txt or has_audio_out:
                    made_progress = True
                    if first_content_ttft_ms is None:
                        first_content_ttft_ms = (time.monotonic() - req_start) * 1000.0
                    if output_txt:
                        est_completion_chars += len(output_txt)
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    got_usage_count = False
                    pt = _safe_nonneg_int(usage.get("prompt_tokens"))
                    if pt is not None:
                        prompt_tokens = pt
                        saw_explicit_prompt_usage = True
                        got_usage_count = True
                        made_progress = True
                    ct = _safe_nonneg_int(usage.get("completion_tokens"))
                    if ct is not None:
                        completion_tokens = ct
                        saw_explicit_completion_usage = True
                        got_usage_count = True
                        made_progress = True
                    has_choices = isinstance(choices, list) and len(choices) > 0
                    if got_usage_count and not has_choices:
                        saw_final_usage = True
                timings = obj.get("timings")
                if isinstance(timings, dict):
                    if not saw_explicit_prompt_usage:
                        pt = _safe_nonneg_int(timings.get("prompt_n"))
                        if pt is not None:
                            prompt_tokens = pt
                    if not saw_explicit_completion_usage:
                        ct = _safe_nonneg_int(timings.get("predicted_n"))
                        if ct is not None:
                            completion_tokens = ct
                            saw_timings_completion = True
                return made_progress, False

            def process_sse_event() -> bool:
                nonlocal current_event_type
                ev_type = current_event_type
                current_event_type = ""
                made_progress, _ = process_sse_event_from(ev_type, event_data_lines)
                event_data_lines.clear()
                return made_progress

            def inspect_sse_line(raw_line: bytes) -> bool:
                nonlocal current_event_type
                if raw_line in (b"\n", b"\r\n", b"\r"):
                    return process_sse_event()
                content_line = raw_line.rstrip(b"\r\n")
                if content_line.startswith(b"event:"):
                    current_event_type = (
                        content_line[6:].strip().decode("utf-8", errors="replace")
                    )
                elif content_line.startswith(b"data:"):
                    data_val = content_line[5:]
                    if data_val.startswith(b" "):
                        data_val = data_val[1:]
                    event_data_lines.append(data_val)
                return False

            def _supports_chunked_transfer(version: str) -> bool:
                if not version.startswith("HTTP/"):
                    return False
                parts = version[5:].split(".", 1)
                if len(parts) != 2:
                    return False
                try:
                    return (int(parts[0]), int(parts[1])) >= (1, 1)
                except ValueError:
                    return False

            use_chunked_downstream = _supports_chunked_transfer(
                str(getattr(self, "request_version", "HTTP/1.1"))
            )

            def write_chunk(data: bytes | bytearray | memoryview) -> None:
                if not data:
                    return
                if use_chunked_downstream:
                    self.connection.sendall(f"{len(data):x}\r\n".encode("ascii"))
                    self.connection.sendall(data)
                    self.connection.sendall(b"\r\n")
                else:
                    self.connection.sendall(data)

            self.close_connection = True
            _, acao = self._resolve_cors_header()
            if committed_flag is not None:
                committed_flag[0] = True
            try:
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type",
                    resp.getheader("Content-Type") or "text/event-stream; charset=utf-8",
                )
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                if use_chunked_downstream:
                    self.send_header("Transfer-Encoding", "chunked")
                self._send_cors_headers(acao)
                self.send_header("Connection", "close")
                self.send_header("X-FreeTPU-Backend", backend.config.id)
                self.send_header("X-FreeTPU-Failovers", str(failovers))
                self.send_header("X-FreeTPU-Request-Id", req_id)
                self.end_headers()

                scan = 0
                pre_len = len(pre_commit_buf)
                while scan < pre_len:
                    nl = pre_commit_buf.find(b"\n", scan, pre_len)
                    cr = pre_commit_buf.find(b"\r", scan, pre_len)
                    if nl == -1 and cr == -1:
                        ln_end = pre_len
                    elif cr != -1 and (nl == -1 or cr < nl):
                        ln_end = (
                            cr + 2
                            if (cr + 1 < pre_len and pre_commit_buf[cr + 1] == 10)
                            else cr + 1
                        )
                    else:
                        ln_end = nl + 1
                    inspect_sse_line(bytes(pre_commit_buf[scan:ln_end]))
                    scan = ln_end
                write_chunk(pre_commit_buf)
            except (BrokenPipeError, ConnectionResetError, OSError):
                downstream_write_error = True
            finally:
                # Immediately drop pre-commit buffer and release its shared memory reservation.
                flushed_pre_bytes = pre_commit_bytes
                pre_commit_buf.clear()
                pre_commit_bytes = 0
                release_sse_mem(flushed_pre_bytes)

            if not downstream_write_error and not saw_done_sentinel:
                if isinstance(upstream_sock, DeadlineSocket):
                    upstream_sock._check_client_on_fast_read = False
                progress_deadline = time.monotonic() + timeout_s
                max_event_bytes = engine.config.max_sse_pre_commit_bytes
                rem_bytes, phase2_eof = sse_reader.drain_buffer()

                raw_tcp_sock = (
                    upstream_sock._sock
                    if isinstance(upstream_sock, DeadlineSocket)
                    else upstream_sock
                )
                reactor = getattr(self.server, "sse_reactor", None)
                use_reactor = (
                    not phase2_eof
                    and reactor is not None
                    and isinstance(raw_tcp_sock, socket.socket)
                    and not isinstance(raw_tcp_sock, ssl.SSLSocket)
                    and isinstance(self.connection, socket.socket)
                    and not isinstance(self.connection, ssl.SSLSocket)
                    and not resp.isclosed()
                )
                if use_reactor:
                    fp_extra = b""
                    fp = getattr(resp, "fp", None)
                    if fp is not None and hasattr(fp, "peek"):
                        if isinstance(upstream_sock, DeadlineSocket):
                            upstream_sock._buffered_peek_only = True
                        try:
                            peeked = fp.peek()
                            if peeked:
                                fp_extra = fp.read1(len(peeked))
                        except ClientDisconnectedError:
                            downstream_write_error = True
                            use_reactor = False
                        except OSError:
                            fp_extra = b""
                        finally:
                            if isinstance(upstream_sock, DeadlineSocket):
                                upstream_sock._buffered_peek_only = False
                    if (
                        use_reactor
                        and not resp.chunked
                        and resp.status not in (204, 304)
                        and not (100 <= resp.status < 200)
                        and resp.length is not None
                    ):
                        fp_extra = fp_extra[: max(0, resp.length)]
                    if use_reactor and fp_extra:
                        if not engine.try_reserve_response_memory(len(fp_extra)):
                            memory_limit_exceeded = True
                            use_reactor = False
                        else:
                            sse_mem_box[0] += len(fp_extra)

                    if use_reactor:
                        if resp.chunked:
                            sse_buf = bytearray(rem_bytes)
                            raw_buf = bytearray(fp_extra)
                            rem_len = None
                        else:
                            sse_buf = bytearray(rem_bytes)
                            if fp_extra:
                                sse_buf.extend(fp_extra)
                            raw_buf = bytearray()
                            if (
                                resp.status not in (204, 304)
                                and not (100 <= resp.status < 200)
                                and resp.length is not None
                            ):
                                rem_len = max(0, resp.length - len(fp_extra))
                            else:
                                rem_len = None

                        task = _SSEReactorTask(
                            upstream_sock=raw_tcp_sock,
                            client_sock=self.connection,
                            engine=engine,
                            sse_buf=sse_buf,
                            raw_buf=raw_buf,
                            chunked=bool(resp.chunked),
                            chunk_left=resp.chunk_left if resp.chunked else None,
                            remaining_content_length=rem_len,
                            max_event_bytes=max_event_bytes,
                            timeout_s=timeout_s,
                            progress_deadline=progress_deadline,
                            sse_mem_box=sse_mem_box,
                            on_event_fn=process_sse_event_from,
                            event_data_lines=event_data_lines,
                            current_event_type=current_event_type,
                            chunked_downstream=use_chunked_downstream,
                            skip_leading_lf=pre_commit_skip_leading_lf,
                        )
                        reactor.submit(task)
                        task.done_event.wait()
                        saw_done_sentinel = task.saw_done_sentinel
                        upstream_read_error = task.upstream_read_error
                        downstream_write_error = task.downstream_write_error
                        size_limit_exceeded = task.size_limit_exceeded
                        memory_limit_exceeded = task.memory_limit_exceeded
                if not use_reactor and not memory_limit_exceeded and not downstream_write_error:
                    if (
                        not resp.chunked
                        and resp.status not in (204, 304)
                        and not (100 <= resp.status < 200)
                        and resp.length is not None
                        and resp.length <= 0
                    ):
                        phase2_eof = True
                    phase2_buf = bytearray(rem_bytes)
                    scan_pos = 0
                    event_start = 0
                    phase2_skip_leading_lf = pre_commit_skip_leading_lf
                    while True:
                        flush_end = 0
                        while True:
                            if phase2_skip_leading_lf:
                                if scan_pos >= len(phase2_buf):
                                    break
                                phase2_skip_leading_lf = False
                                if phase2_buf[scan_pos] == 10:
                                    was_event_start = event_start == scan_pos
                                    scan_pos += 1
                                    if was_event_start:
                                        event_start = scan_pos
                                        flush_end = scan_pos
                                    continue
                            nl_idx = phase2_buf.find(b"\n", scan_pos)
                            cr_idx = phase2_buf.find(b"\r", scan_pos)
                            if nl_idx == -1 and cr_idx == -1:
                                if len(phase2_buf) - event_start > max_event_bytes:
                                    size_limit_exceeded = True
                                break
                            if cr_idx != -1 and (nl_idx == -1 or cr_idx < nl_idx):
                                content_end = cr_idx
                                if cr_idx + 1 < len(phase2_buf) and phase2_buf[cr_idx + 1] == 10:
                                    line_end = cr_idx + 2
                                else:
                                    line_end = cr_idx + 1
                                    if line_end == len(phase2_buf):
                                        phase2_skip_leading_lf = True
                            else:
                                content_end = nl_idx
                                line_end = nl_idx + 1
                            if line_end - event_start > max_event_bytes:
                                size_limit_exceeded = True
                                break
                            if content_end == scan_pos:
                                if process_sse_event():
                                    progress_deadline = time.monotonic() + timeout_s
                                event_start = line_end
                                flush_end = line_end
                                scan_pos = line_end
                                if saw_done_sentinel:
                                    break
                            else:
                                if phase2_buf.startswith(b"data:", scan_pos):
                                    data_start = scan_pos + 5
                                    if data_start < content_end and phase2_buf[data_start] == 32:
                                        data_start += 1
                                    event_data_lines.append(bytes(phase2_buf[data_start:content_end]))
                                elif phase2_buf.startswith(b"event:", scan_pos):
                                    current_event_type = (
                                        bytes(phase2_buf[scan_pos + 6 : content_end])
                                        .strip()
                                        .decode("utf-8", errors="replace")
                                    )
                                scan_pos = line_end

                        if flush_end > 0:
                            try:
                                write_chunk(bytes(phase2_buf[:flush_end]))
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                downstream_write_error = True
                                break
                            finally:
                                del phase2_buf[:flush_end]
                                scan_pos -= flush_end
                                event_start -= flush_end
                                release_sse_mem(flush_end)

                        if saw_done_sentinel or size_limit_exceeded or phase2_eof:
                            break

                        rem = progress_deadline - time.monotonic()
                        if rem <= 0:
                            upstream_read_error = True
                            break
                        if isinstance(upstream_sock, DeadlineSocket):
                            upstream_sock.deadline = progress_deadline
                        elif upstream_sock is not None:
                            try:
                                upstream_sock.settimeout(rem)
                            except OSError:
                                pass

                        avail = engine.available_response_memory()
                        rem_ev_cap = (max_event_bytes - (len(phase2_buf) - event_start)) + 1
                        to_read = min(65536, max(1, avail), max(1, rem_ev_cap))
                        try:
                            chunk = resp.read1(to_read)
                        except ClientDisconnectedError:
                            downstream_write_error = True
                            break
                        except (OSError, TimeoutError, http.client.HTTPException):
                            upstream_read_error = True
                            break

                        if not chunk:
                            if (
                                not resp.chunked
                                and resp.status not in (204, 304)
                                and not (100 <= resp.status < 200)
                                and resp.length is not None
                                and resp.length > 0
                            ):
                                upstream_read_error = True
                            phase2_eof = True
                            break

                        if not engine.try_reserve_response_memory(len(chunk)):
                            memory_limit_exceeded = True
                            break
                        sse_mem_box[0] += len(chunk)
                        phase2_buf.extend(chunk)

                    if phase2_buf:
                        rem_ev = len(phase2_buf)
                        phase2_buf.clear()
                        release_sse_mem(rem_ev)

            # Per WHATWG SSE spec, discard any incomplete (unterminated) event data at EOF
            # rather than letting it satisfy completion checks.
            all_choices_finished = all(
                idx in finished_choice_indices for idx in range(required_choices)
            )
            usage_satisfied = (not expect_final_usage) or saw_final_usage
            stream_completed = (
                not saw_upstream_error_event
                and (saw_done_sentinel or (all_choices_finished and usage_satisfied))
            )
            stream_memory_exceeded = not downstream_write_error and memory_limit_exceeded
            stream_size_exceeded = (
                not downstream_write_error
                and not stream_memory_exceeded
                and size_limit_exceeded
            )
            premature_upstream_end = (
                not downstream_write_error
                and not stream_memory_exceeded
                and not stream_size_exceeded
                and (upstream_read_error or not stream_completed)
            )
            if (
                not saw_explicit_completion_usage
                and not saw_timings_completion
                and est_completion_chars > 0
            ):
                completion_tokens = max(1, (est_completion_chars + 3) // 4)
            total_ms = (time.monotonic() - req_start) * 1000.0
            attempt_ms = (time.monotonic() - attempt_start) * 1000.0
            effective_ttft = (
                first_content_ttft_ms if first_content_ttft_ms is not None else ttft_ms
            )
            if downstream_write_error:
                # Downstream client disconnected or stalled on writes: record a distinct
                # client-closed status (499) in Prometheus and UsageMeterHook while
                # retaining partial token counts, without updating backend success latency
                # or penalizing backend health.
                recorded_status = 499
            elif stream_memory_exceeded:
                # Shared memory exhaustion is request-scoped: do not penalize backend health.
                recorded_status = 503
            elif stream_size_exceeded:
                # Local event size cap is request-scoped: do not penalize backend health.
                recorded_status = 502
            elif premature_upstream_end:
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
                    if stream_memory_exceeded:
                        err_event = (
                            b"event: error\n"
                            b'data: {"error":{"message":"Router response memory budget exhausted; retry later",'
                            b'"type":"service_unavailable","code":"server_memory_limit_reached"}}\n\n'
                        )
                        write_chunk(err_event)
                        self.wfile.flush()
                    elif stream_size_exceeded:
                        err_event = (
                            b"event: error\n"
                            b'data: {"error":{"message":"Upstream SSE event exceeded maximum size limit",'
                            b'"type":"bad_gateway","code":"response_too_large"}}\n\n'
                        )
                        write_chunk(err_event)
                        self.wfile.flush()
                    elif premature_upstream_end:
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
                        if use_chunked_downstream:
                            self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    downstream_write_error = True

            return "committed", recorded_status, None
        finally:
            pre_commit_buf.clear()
            if sse_mem_box[0] > 0:
                engine.release_response_memory(sse_mem_box[0])
                sse_mem_box[0] = 0


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
        ),
        allow_abbrev=False,
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
        "--bearer-token-env",
        action="append",
        dest="bearer_token_envs",
        help=(
            "Environment variable name holding a bearer token required on API "
            "requests (repeatable; avoids exposing live tokens in process argv)."
        ),
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


def _apply_cli_overrides(cfg: RouterConfig, args: argparse.Namespace) -> None:
    """Apply explicit CLI flag overrides onto a loaded `RouterConfig`."""
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.policy is not None:
        cfg.policy = args.policy
    if args.bearer_token_envs:
        existing_auth = cfg.auth if isinstance(cfg.auth, Mapping) else {}
        new_auth: Dict[str, Any] = {
            "type": "bearer",
            "tokens_env": list(args.bearer_token_envs),
        }
        if "exempt_paths" in existing_auth:
            new_auth["exempt_paths"] = existing_auth["exempt_paths"]
        cfg.auth = new_auth
    if args.cors_origins:
        cfg.cors_allowed_origins = list(args.cors_origins)
    if args.usage_jsonl:
        cfg.usage_meter = {"type": "jsonl", "jsonl_path": str(args.usage_jsonl)}


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
        _apply_cli_overrides(cfg, args)
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
