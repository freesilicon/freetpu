#!/usr/bin/env python3
"""
Clean-room headless Google Colab TPU benchmark and probe harness for FreeTPU.

Wraps the sanctioned `google-colab-cli` (`colab`) control plane to run
reproducible, zero-idle-waste TPU probes (`v6e1` Trillium with graceful
fallback to `v5e1`):

  1. Private owner-only output/config directory validation (no symlinks or shared /tmp races).
  2. Pre-flight signed Compute Unit (CU) balance & active-session snapshot via `colab usage` / `colab sessions`.
  3. Per-run normalized session state isolation (`--config`) and unique session names.
  4. Ephemeral provision -> execute -> auto-stop via `colab run -- ...` with post-run `colab sessions` release verification.
  5. Conservative `v6e1` -> `v5e1` fallback ONLY on confirmed allocation refusal (never on ReadTimeout or post-READY failures).
  6. Replayable notebook proof export via `colab log -s <session> -o <path>.ipynb`.
  7. Offline `--self-test` and `--dry-run` modes so CI passes without Colab credentials or live hardware.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PROBE_JSON_BEGIN = "=== FREETPU_COLAB_PROBE_JSON_BEGIN ==="
PROBE_JSON_END = "=== FREETPU_COLAB_PROBE_JSON_END ==="

DEFAULT_TPU_ORDER: Tuple[str, ...] = ("v6e1", "v5e1")
SUPPORTED_TPUS = frozenset({"v6e1", "v5e1"})
DEFAULT_REQUEST_TIMEOUT_S: float = 60.0
DEFAULT_STARTUP_OVERHEAD_S: float = 180.0
DEFAULT_UNBOUNDED_INACTIVITY_WALL_TIMEOUT_S: float = 480.0
DEFAULT_MAX_CAPTURE_BYTES: int = 256 * 1024
DEFAULT_MAX_PROBE_CAPTURE_BYTES: int = 256 * 1024
MIN_COLAB_CLI_VERSION: Tuple[int, int, int] = (0, 7, 6)
_VERSION_RE = re.compile(
    r"\b(\d+)\.(\d+)\.(\d+)(?:(?:\.?post\d+)|([.-]?(?:a|b|rc|alpha|beta|dev|pre)\w*))?",
    re.IGNORECASE,
)

_ALLOCATION_DIAGNOSTIC_PATTERNS = (
    re.compile(r"^\[colab\]\s+allocation refused\b"),
    re.compile(
        r"^\[colab\]\s+.*\b(reason_no_backends|reason_too_many_active_assignments|"
        r"reason_quota_exceeded_usage_time|reason_runtime_version_unavailable|"
        r"failed_precondition|precondition failed|quota or entitlement|"
        r"too many active sessions|no backends|backend rejected accelerator)\b"
    ),
    re.compile(r"^(?:[a-z_][a-z0-9_.]*\.)?toomanyassignmentserror(?:\s*:|\s*$)"),
    re.compile(
        r"^(?:error:\s*)?.*\b(allocation refused|backend rejected accelerator|"
        r"reason_no_backends|reason_too_many_active_assignments|"
        r"reason_quota_exceeded_usage_time|reason_runtime_version_unavailable|"
        r"too many active sessions)\b"
    ),
)

_TRACEBACK_CODE_LINE_RE = re.compile(
    r"^(?:[│╭╰❱─]|traceback\s*\(|file\s+\"|\d+\s+[│|]|"
    r"(?:except|raise|try|def|class|if|elif|else|for|while|with|return|from|import)\b)"
)

_OAUTH_PROMPT_MARKERS = (
    "to authorize colab-cli",
    "enter the authorization code",
    "eoferror",
)

_COLAB_AUTH_ERROR_MARKERS = (
    "no valid default credentials found",
    "to authorize colab-cli",
    "enter the authorization code",
    "defaultcredentialserror",
    "could not automatically determine credentials",
    "reauthentication is needed",
    "failed to refresh token",
    "invalid_grant",
)

_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_SAFE_STORED_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_ISO_DATETIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[Tt ](?P<hm>\d{2}:\d{2})"
    r"(?::(?P<sec>\d{2})(?:\.(?P<frac>\d{1,9}))?)?"
    r"(?P<tz>[Zz]|(?P<tz_sign>[+-])(?P<tz_h>\d{2}):?(?P<tz_m>\d{2}))?\Z",
    re.ASCII,
)
_SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\S+)(?:\s+\||$)")


class UntrustedPathError(ValueError):
    """Raised when an output or config path is symlinked or not private to the current user."""


class TerminationSignal(BaseException):
    """Raised when SIGTERM is delivered so active TPU sessions are reconciled before exit."""


class CommandCancelledError(BaseException):
    """Raised when a running subprocess is interrupted, preserving stdout/stderr emitted so far."""

    def __init__(
        self,
        *,
        signal_name: str,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        stdout_log_written: bool = False,
        stderr_log_written: bool = False,
        log_write_error: Optional[str] = None,
        untrusted_log_error: Optional[str] = None,
        captured_probe_text: Optional[str] = None,
        elapsed_s: Optional[float] = None,
    ) -> None:
        super().__init__(f"Command cancelled by {signal_name}")
        self.signal_name = signal_name
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_log_written = stdout_log_written
        self.stderr_log_written = stderr_log_written
        self.log_write_error = log_write_error
        self.untrusted_log_error = untrusted_log_error
        self.captured_probe_text = captured_probe_text
        self.elapsed_s = elapsed_s


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_log_written: bool = False
    stderr_log_written: bool = False
    log_write_error: Optional[str] = None
    untrusted_log_error: Optional[str] = None
    captured_probe_text: Optional[str] = None
    elapsed_s: Optional[float] = None
    pre_ready_uncertain: bool = False


CommandRunner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class UsageSnapshot:
    balance_cu: Optional[float]
    usage_rate_per_hour: Optional[float]
    active_assignments: Optional[int]
    raw_output: str


@dataclass(frozen=True)
class BenchConfig:
    colab_bin: str = "colab"
    auth: str = "adc"
    tpus: Tuple[str, ...] = DEFAULT_TPU_ORDER
    min_cu_balance: float = 5.0
    max_active_assignments: Optional[int] = None
    timeout_s: float = 300.0
    wall_timeout_s: Optional[float] = None
    request_timeout_s: Optional[float] = None
    teardown_retries: int = 2
    teardown_retry_delay_s: float = 2.0
    use_no_public_api: bool = True
    config_path: Optional[Path] = None
    out_dir: Optional[Path] = None
    script_path: Optional[Path] = None
    script_args: Tuple[str, ...] = ()
    session_name: Optional[str] = None
    dry_run: bool = False


_MAX_SYMLINK_HOPS = 40


def _validate_dir_stat(path: Path, st: os.stat_result) -> None:
    """Validate that `st` describes a safe traversed ancestor directory."""
    if not stat.S_ISDIR(st.st_mode):
        raise UntrustedPathError(f"Refusing non-directory ancestor {path}")
    euid = os.geteuid() if hasattr(os, "geteuid") else st.st_uid
    if st.st_uid != euid and st.st_uid != 0:
        raise UntrustedPathError(
            f"Refusing path under unowned ancestor {path} (uid={st.st_uid})"
        )
    # If world/group-writable, require sticky bit (e.g., /tmp, /var/tmp) so other users cannot
    # replace or unlink subdirectories created inside it.
    if (st.st_mode & 0o022) != 0 and not (st.st_mode & stat.S_ISVTX):
        raise UntrustedPathError(
            f"Refusing path under untrusted writable ancestor {path} (mode={oct(stat.S_IMODE(st.st_mode))})"
        )


def _resolve_and_validate_existing_dir(start_dir: Path) -> Path:
    """Walk `start_dir` component-by-component, validating every intermediate directory and symlink.

    Unlike `os.path.realpath`, this expands each symlink one hop at a time via `os.readlink`
    so intermediate directories (e.g. a collaborator-writable scratch directory reached via an
    intermediate symlink) are never skipped. Returns the validated canonical directory path.
    """
    raw = start_dir if start_dir.is_absolute() else Path(os.getcwd()) / start_dir
    anchor = Path(raw.anchor or "/")
    try:
        root_st = os.lstat(anchor)
    except OSError as exc:
        raise UntrustedPathError(f"Cannot stat filesystem root {anchor}: {exc}") from exc
    _validate_dir_stat(anchor, root_st)

    resolved = anchor
    pending: deque[str] = deque(raw.parts[1:])
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    hops = 0

    while pending:
        part = pending.popleft()
        if not part or part == ".":
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        try:
            st = os.lstat(candidate)
        except OSError as exc:
            raise UntrustedPathError(f"Cannot stat path component {candidate}: {exc}") from exc

        if stat.S_ISLNK(st.st_mode):
            hops += 1
            if hops > _MAX_SYMLINK_HOPS:
                raise UntrustedPathError(f"Too many symbolic links while resolving {start_dir}")
            if euid is not None and st.st_uid != euid and st.st_uid != 0:
                raise UntrustedPathError(
                    f"Refusing path through unowned symlink {candidate} (uid={st.st_uid})"
                )
            try:
                target = os.readlink(candidate)
            except OSError as exc:
                raise UntrustedPathError(f"Cannot read symlink {candidate}: {exc}") from exc
            target_path = Path(target)
            if target_path.is_absolute():
                resolved = Path(target_path.anchor or "/")
                for p in reversed(target_path.parts[1:]):
                    pending.appendleft(p)
            else:
                for p in reversed(target_path.parts):
                    pending.appendleft(p)
            continue

        _validate_dir_stat(candidate, st)
        resolved = candidate

    return resolved


def _ensure_private_dir(raw_path: Path) -> Path:
    """Ensure `raw_path` exists as a canonical non-symlink directory owned by the user (mode 0700)."""
    expanded = Path(os.path.expanduser(str(raw_path)))
    if expanded.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked directory path: {raw_path}")

    raw_abs = expanded if expanded.is_absolute() else Path(os.getcwd()) / expanded
    if os.path.lexists(raw_abs):
        if os.path.islink(raw_abs):
            raise UntrustedPathError(f"Refusing symlinked directory: {raw_abs}")
        if raw_abs.name in ("", ".", ".."):
            canonical_target = _resolve_and_validate_existing_dir(raw_abs)
        else:
            canonical_parent = _resolve_and_validate_existing_dir(raw_abs.parent)
            canonical_target = canonical_parent / raw_abs.name
    else:
        missing_names: List[str] = []
        cur = raw_abs
        while not os.path.lexists(cur):
            if cur.name in ("", ".", ".."):
                raise UntrustedPathError(f"Refusing invalid missing path component in {raw_path}")
            missing_names.append(cur.name)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        canonical_cur = _resolve_and_validate_existing_dir(cur)
        for name in reversed(missing_names):
            canonical_cur = canonical_cur / name
            try:
                os.mkdir(canonical_cur, 0o700)
            except FileExistsError:
                pass
        canonical_target = canonical_cur

    try:
        st = os.lstat(canonical_target)
    except OSError as exc:
        raise UntrustedPathError(f"Cannot stat directory {canonical_target}: {exc}") from exc

    if stat.S_ISLNK(st.st_mode):
        raise UntrustedPathError(f"Refusing symlinked directory: {canonical_target}")
    if not stat.S_ISDIR(st.st_mode):
        raise UntrustedPathError(f"Expected directory, found non-directory: {canonical_target}")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UntrustedPathError(
            f"Directory {canonical_target} is not owned by current user (uid={st.st_uid})"
        )
    if (st.st_mode & 0o077) != 0:
        raise UntrustedPathError(
            f"Directory {canonical_target} must be private (mode 0700); got {oct(stat.S_IMODE(st.st_mode))}"
        )
    return canonical_target


def _write_text_nofollow(path: Path, text: str) -> None:
    """Write UTF-8 text via a private temp file and atomically replace `path` (mode 0600)."""
    if path.is_symlink():
        raise UntrustedPathError(f"Refusing to write through symlink: {path}")
    canonical_parent = _ensure_dir_not_shared_writable(path.parent)
    target = canonical_parent / path.name
    if target.is_symlink():
        raise UntrustedPathError(f"Refusing to write through symlink: {target}")
    if os.path.lexists(target):
        st = os.lstat(target)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise UntrustedPathError(f"Refusing unsafe existing target file {target}")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(canonical_parent),
    )
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise UntrustedPathError(f"Target is not a regular file: {tmp_path}")
            if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
                raise UntrustedPathError(f"Target file {tmp_path} has unexpected owner uid={st.st_uid}")
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _ensure_dir_not_shared_writable(dir_path: Path) -> Path:
    """Ensure `dir_path` exists as a canonical operator-owned non-symlink dir not writable by others."""
    expanded = Path(os.path.expanduser(str(dir_path)))
    if expanded.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked directory path: {dir_path}")
    raw_abs = expanded if expanded.is_absolute() else Path(os.getcwd()) / expanded
    if not os.path.lexists(raw_abs):
        return _ensure_private_dir(raw_abs)
    if os.path.islink(raw_abs):
        raise UntrustedPathError(f"Refusing symlinked directory path: {raw_abs}")
    if raw_abs.name in ("", ".", ".."):
        canonical_dir = _resolve_and_validate_existing_dir(raw_abs)
    else:
        canonical_parent = _resolve_and_validate_existing_dir(raw_abs.parent)
        canonical_dir = canonical_parent / raw_abs.name
    st = os.lstat(canonical_dir)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise UntrustedPathError(f"Refusing non-directory/symlink config parent: {canonical_dir}")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UntrustedPathError(
            f"Config parent directory {canonical_dir} is not owned by current user (uid={st.st_uid})"
        )
    if (st.st_mode & 0o022) != 0:
        raise UntrustedPathError(
            f"Config parent directory {canonical_dir} must not be group/world-writable (mode={oct(stat.S_IMODE(st.st_mode))})"
        )
    return canonical_dir


def _verify_file_writable_if_exists(path: Path, label: str) -> None:
    """Verify an existing file is a user-owned, non-shared-writable regular non-symlink file."""
    if not os.path.lexists(path):
        return
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise UntrustedPathError(f"Cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UntrustedPathError(f"Refusing symlinked {label}: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise UntrustedPathError(f"Refusing non-regular {label}: {path}")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UntrustedPathError(
            f"{label} {path} is not owned by current user (uid={st.st_uid})"
        )
    if (st.st_mode & 0o022) != 0:
        raise UntrustedPathError(
            f"{label} {path} must not be group/world-writable (mode={oct(stat.S_IMODE(st.st_mode))})"
        )
    if st.st_nlink > 1:
        raise UntrustedPathError(
            f"Refusing hard-linked {label} {path} (st_nlink={st.st_nlink})"
        )
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise UntrustedPathError(f"{label} {path} is not writable: {exc}") from exc
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            raise UntrustedPathError(f"Refusing non-regular {label}: {path}")
        if hasattr(os, "geteuid") and fst.st_uid != os.geteuid():
            raise UntrustedPathError(
                f"{label} {path} is not owned by current user (uid={fst.st_uid})"
            )
        if (fst.st_mode & 0o022) != 0:
            raise UntrustedPathError(
                f"{label} {path} must not be group/world-writable (mode={oct(stat.S_IMODE(fst.st_mode))})"
            )
        if fst.st_nlink > 1:
            raise UntrustedPathError(
                f"Refusing hard-linked {label} {path} (st_nlink={fst.st_nlink})"
            )
    finally:
        os.close(fd)


def _is_valid_iso_datetime(value: Any) -> bool:
    """Return True if `value` matches Pydantic/speedate RFC-3339 datetime validation."""
    if not isinstance(value, str):
        return False
    m = _ISO_DATETIME_RE.fullmatch(value)
    if not m:
        return False
    tz_part = ""
    if m.group("tz"):
        if m.group("tz") in ("Z", "z"):
            tz_part = "+00:00"
        else:
            tz_h = int(m.group("tz_h"))
            tz_m = int(m.group("tz_m"))
            if tz_h > 23 or tz_m > 59:
                return False
            tz_part = f"{m.group('tz_sign')}{tz_h:02d}:{tz_m:02d}"
    time_part = m.group("hm")
    if m.group("sec") is not None:
        time_part += f":{m.group('sec')}"
        if m.group("frac") is not None:
            time_part += f".{m.group('frac')[:6]}"
    try:
        datetime.fromisoformat(f"{m.group('date')}T{time_part}{tz_part}")
    except ValueError:
        return False
    return True


def _validate_existing_session_store(path: Path) -> bool:
    """Validate that an existing non-empty `--config` file matches Colab's full `SessionState` schema.

    Returns True iff the file exists and already contains one or more session mappings.
    Colab CLI v0.8.0 treats any deserialization failure on a single entry as an empty store and
    truncates the whole file on the next write; fail closed here before calling Colab.
    Never includes file contents in error messages so no tokens or metadata can be leaked.
    """
    if not os.path.lexists(path):
        return False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise UntrustedPathError(f"Cannot read session config file {path}: {exc}") from exc
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            raise UntrustedPathError(f"Refusing non-regular session config file: {path}")
        if fst.st_nlink > 1:
            raise UntrustedPathError(
                f"Refusing hard-linked session config file {path} (st_nlink={fst.st_nlink})"
            )
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            raw = fh.read()
    except UntrustedPathError:
        raise
    except OSError as exc:
        raise UntrustedPathError(f"Cannot read session config file {path}: {exc}") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    if not raw.strip():
        return False
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UntrustedPathError(
            f"Existing session config {path} is not a valid Colab session store JSON file."
        ) from exc
    if not isinstance(parsed, dict):
        raise UntrustedPathError(
            f"Existing session config {path} must be a JSON object mapping session names to entries."
        )
    for key, val in parsed.items():
        if (
            not isinstance(key, str)
            or not key
            or ".." in key
            or not _SAFE_STORED_SESSION_RE.fullmatch(key)
            or not isinstance(val, dict)
        ):
            raise UntrustedPathError(
                f"Existing session config {path} contains an invalid or unsafe session store entry."
            )
        if (
            not isinstance(val.get("name"), str)
            or val.get("name") != key
            or not isinstance(val.get("token"), str)
            or not isinstance(val.get("url"), str)
            or not isinstance(val.get("endpoint"), str)
            or not val.get("endpoint")
        ):
            raise UntrustedPathError(
                f"Existing session config {path} does not match the Colab session store schema."
            )
        # Optional non-nullable string fields in `colab_cli.state.SessionState`.
        for str_field in ("variant", "accelerator", "machine_shape"):
            if str_field in val and not isinstance(val[str_field], str):
                raise UntrustedPathError(
                    f"Existing session config {path} has invalid `{str_field}` field type."
                )
        # Optional nullable string fields in `colab_cli.state.SessionState`.
        for opt_str_field in ("kernel_id", "session_id", "running"):
            if opt_str_field in val and val[opt_str_field] is not None and not isinstance(
                val[opt_str_field], str
            ):
                raise UntrustedPathError(
                    f"Existing session config {path} has invalid `{opt_str_field}` field type."
                )
        # Optional ISO-8601 datetime `token_expires_at: Optional[datetime]`.
        if "token_expires_at" in val and val["token_expires_at"] is not None:
            if not _is_valid_iso_datetime(val["token_expires_at"]):
                raise UntrustedPathError(
                    f"Existing session config {path} has malformed `token_expires_at` timestamp."
                )
        # Optional 3-tuple `last_execution: Optional[Tuple[str, Optional[str], str]]`.
        if "last_execution" in val and val["last_execution"] is not None:
            le = val["last_execution"]
            if (
                not isinstance(le, list)
                or len(le) != 3
                or not isinstance(le[0], str)
                or (le[1] is not None and not isinstance(le[1], str))
                or not isinstance(le[2], str)
            ):
                raise UntrustedPathError(
                    f"Existing session config {path} has malformed `last_execution` metadata."
                )
    return bool(parsed)


def _normalize_config_path(
    config_path: Optional[Path],
    out_dir: Path,
    session_base: str = "run",
) -> Path:
    """Normalize explicit or default per-run `--config` to a canonical writable non-symlink path."""
    if config_path is None:
        raw_cfg = out_dir / f"colab-sessions.{session_base}.json"
    else:
        expanded = Path(os.path.expanduser(str(config_path)))
        if expanded.is_symlink():
            raise UntrustedPathError(f"Refusing symlinked --config path: {config_path}")
        raw_cfg = expanded if expanded.is_absolute() else Path(os.getcwd()) / expanded
    if raw_cfg.name in ("", ".", ".."):
        raise UntrustedPathError(f"Refusing invalid --config filename: {config_path}")
    canonical_parent = _ensure_dir_not_shared_writable(raw_cfg.parent)
    if not os.access(canonical_parent, os.W_OK | os.X_OK):
        raise UntrustedPathError(
            f"Config parent directory is not writable: {canonical_parent}"
        )
    canonical_cfg = canonical_parent / raw_cfg.name
    lock_path = Path(f"{canonical_cfg}.lock")
    if canonical_cfg.is_symlink() or lock_path.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked --config or lock path: {canonical_cfg}")
    _verify_file_writable_if_exists(canonical_cfg, "session config file")
    _verify_file_writable_if_exists(lock_path, "session config lock file")
    return canonical_cfg


def _paths_collide(a: Path, b: Path) -> bool:
    if os.path.normcase(str(a)) == os.path.normcase(str(b)):
        return True
    try:
        if os.path.lexists(a) and os.path.lexists(b) and os.path.samefile(a, b):
            return True
    except OSError:
        pass
    return False


def _check_config_artifact_collision(
    isolated_config: Path,
    out_dir: Path,
    session_base: str,
    tpus: Sequence[str],
    script_path: Optional[Path] = None,
) -> Optional[str]:
    """Reject `--config` (or its `.lock`) colliding with generated artifacts or `--script`."""
    candidates: List[Path] = [
        out_dir / "summary.json",
        out_dir / f"{session_base}.summary.json",
        out_dir / "dry-run-plan.json",
        out_dir / "colab_tpu_probe_payload.py",
        out_dir / f"colab_custom_script.{session_base}.py",
        out_dir / f"colab-sessions-list.{session_base}.json",
        out_dir / f"colab-sessions-list.{session_base}.json.lock",
    ]
    for idx, tpu in enumerate(tpus):
        sess = f"{session_base}-{tpu}" if idx > 0 else session_base
        candidates.extend(
            [
                out_dir / f"{sess}.summary.json",
                out_dir / f"{sess}.stdout.txt",
                out_dir / f"{sess}.stderr.txt",
                out_dir / f"{sess}.ipynb",
                out_dir / f"colab-recovery.{sess}.json",
                out_dir / f"colab-recovery.{sess}.json.lock",
            ]
        )
    if script_path is not None:
        expanded_script = Path(os.path.expanduser(str(script_path)))
        candidates.append(expanded_script)
        candidates.append(Path(os.path.realpath(str(expanded_script))))
    lock_path = Path(f"{isolated_config}.lock")
    for check_path, label in ((isolated_config, "--config"), (lock_path, "--config lock")):
        for artifact in candidates:
            if _paths_collide(check_path, artifact):
                return (
                    f"{label} path {check_path} collides with {artifact}; "
                    "choose a distinct session store filename."
                )
        if _paths_collide(check_path.parent, out_dir):
            lower_name = check_path.name.lower()
            if (
                lower_name.startswith("colab-recovery.")
                or lower_name.startswith("colab_custom_script.")
                or lower_name.startswith("colab-sessions-list.")
            ):
                return (
                    f"{label} path {check_path} uses a reserved artifact prefix inside {out_dir}; "
                    "choose a distinct session store filename."
                )
    return None


def _prepare_custom_script(
    script_path: Path,
    out_dir: Path,
    session_base: str,
) -> Tuple[Path, bool]:
    """Validate a custom Python script locally and wrap it so `__file__`, `__future__`, and `__main__` work."""
    try:
        raw_bytes = script_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read custom script {script_path}: {exc}") from exc

    # Compile locally BEFORE provisioning any TPU so syntax errors fail closed with zero CU spent.
    try:
        tree = ast.parse(raw_bytes, filename=str(script_path))
        compile(tree, str(script_path), "exec")
    except SyntaxError as exc:
        raise ValueError(
            f"Custom script {script_path} has a Python syntax error at line {exc.lineno}: {exc.msg}"
        ) from exc

    # Always wrap custom scripts in a thin `__main__` bootstrapper so `colab run`'s injected prelude
    # never breaks `from __future__` / coding cookies and the real source is materialized on disk
    # so `__file__`, `open(__file__)`, and `inspect.getsource()` all work seamlessly at runtime.
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    orig_name = os.path.basename(script_path.name) or "custom_script.py"
    wrapper_src = (
        "import atexit as _ft_atexit, base64 as _ft_b64, os as _ft_os, "
        "shutil as _ft_shutil, sys as _ft_sys, tempfile as _ft_tempfile\n"
        f"_ft_src = _ft_b64.b64decode({encoded!r})\n"
        f"_ft_name = {orig_name!r}\n"
        "_ft_dir = _ft_tempfile.mkdtemp(prefix='freetpu_script_')\n"
        "_ft_os.chmod(_ft_dir, 0o700)\n"
        "_ft_atexit.register(_ft_shutil.rmtree, _ft_dir, True)\n"
        "_ft_path = _ft_os.path.join(_ft_dir, _ft_name)\n"
        "_ft_fd = _ft_os.open(_ft_path, _ft_os.O_WRONLY | _ft_os.O_CREAT | _ft_os.O_EXCL, 0o600)\n"
        "with _ft_os.fdopen(_ft_fd, 'wb') as _ft_fh:\n"
        "    _ft_fh.write(_ft_src)\n"
        "if _ft_sys.argv:\n"
        "    _ft_sys.argv[0] = _ft_path\n"
        "_ft_main = _ft_sys.modules.get('__main__')\n"
        "_ft_ns = _ft_main.__dict__ if _ft_main is not None else globals()\n"
        "_ft_ns['__name__'] = '__main__'\n"
        "_ft_ns['__file__'] = _ft_path\n"
        "_ft_ns['__package__'] = None\n"
        "exec(compile(_ft_src, _ft_path, 'exec'), _ft_ns)\n"
    )
    wrapper_path = out_dir / f"colab_custom_script.{session_base}.py"
    _write_text_nofollow(wrapper_path, wrapper_src)
    return wrapper_path, True


def _coerce_subprocess_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _write_recovery_session_store(
    recovery_config: Path,
    session_name: str,
    endpoint: str,
) -> None:
    """Write a dedicated per-run `colab stop` recovery store without touching shared session state.

    `google-colab-cli` v0.8.0's `colab run` teardown swallows `unassign()` failures and still
    removes the session from local state. Writing a minimal `SessionState`-compatible entry
    with `token_expires_at: null` into a dedicated per-run file lets `colab stop -s <session_name>`
    refresh proxy metadata from `list_assignments()` and unassign the stranded endpoint without
    racing against concurrent workers sharing an explicit `--config` file.
    """
    if recovery_config.is_symlink() or Path(f"{recovery_config}.lock").is_symlink():
        raise UntrustedPathError(
            f"Refusing symlinked recovery config/lock path: {recovery_config}"
        )
    data = {
        session_name: {
            "name": session_name,
            "token": "",
            "url": "",
            "endpoint": endpoint,
            "token_expires_at": None,
        }
    }
    _write_text_nofollow(recovery_config, json.dumps(data, indent=2) + "\n")


def is_supported_colab_cli_version(version_text: str) -> Tuple[bool, Optional[str]]:
    """Return `(supported, parsed_version)` requiring `google-colab-cli >= 0.7.6` (final).

    Releases prior to `0.7.6` logged unredacted authorization headers/tokens to
    `~/.config/colab-cli/colab.log` under the default umask.
    """
    m = _VERSION_RE.search(version_text)
    if not m:
        return False, None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    prerelease = m.group(4)
    parsed_str = m.group(0)
    ver_tuple = (major, minor, patch)
    if ver_tuple < MIN_COLAB_CLI_VERSION:
        return False, parsed_str
    if ver_tuple == MIN_COLAB_CLI_VERSION and prerelease:
        return False, parsed_str
    return True, parsed_str


def check_colab_cli_version(
    colab_bin: str,
    runner: CommandRunner,
) -> Tuple[bool, Optional[str], Optional[str], int]:
    """Run non-authenticated `colab version` and enforce `>= 0.7.6` before any auth command."""
    try:
        res = runner([colab_bin, "version"])
    except FileNotFoundError:
        return False, None, f"Executable {colab_bin!r} not found in PATH.", 127
    combined = f"{res.stdout}\n{res.stderr}".strip()
    if res.returncode != 0:
        return (
            False,
            None,
            (
                f"`{colab_bin} version` failed (rc={res.returncode}): "
                f"{combined or 'no diagnostic output'}. Upgrade to google-colab-cli >= 0.7.6."
            ),
            res.returncode,
        )
    supported, parsed = is_supported_colab_cli_version(combined)
    if not supported:
        min_str = ".".join(str(x) for x in MIN_COLAB_CLI_VERSION)
        return (
            False,
            parsed,
            (
                f"Unsupported google-colab-cli version {parsed or combined!r}; "
                f"version >= {min_str} is required for redacted/private credential logging."
            ),
            res.returncode,
        )
    return True, parsed, None, res.returncode


def resolve_effective_wall_timeout(
    timeout_s: float,
    wall_timeout_s: Optional[float],
) -> Optional[float]:
    """Compute the effective local wall-clock deadline for `colab run`.

    - Explicit `wall_timeout_s == 0`: opt-out (unbounded local wall time).
    - Explicit `wall_timeout_s > 0`: exact local wall-clock cap.
    - Omitted (`wall_timeout_s is None`): finite default (`timeout_s + 180s` when `timeout_s > 0`,
      or `480s` when `timeout_s == 0`) so a stalled kernel setup cell cannot leak Compute Units.
    """
    if wall_timeout_s is not None:
        return wall_timeout_s if wall_timeout_s > 0 else None
    if timeout_s > 0:
        return timeout_s + DEFAULT_STARTUP_OVERHEAD_S
    return DEFAULT_UNBOUNDED_INACTIVITY_WALL_TIMEOUT_S


def resolve_effective_request_timeout(
    request_timeout_s: Optional[float],
) -> Tuple[float, bool]:
    """Return `(effective_request_timeout_s, explicit_override)` honoring explicit flag over env."""
    if request_timeout_s is not None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError(
                f"`--request-timeout` must be finite and > 0 (got {request_timeout_s!r})."
            )
        return request_timeout_s, True
    raw_env = os.environ.get("COLAB_CLI_REQUEST_TIMEOUT")
    if raw_env is not None and raw_env.strip():
        try:
            env_val = float(raw_env.strip())
        except ValueError as exc:
            raise ValueError(
                f"Inherited `COLAB_CLI_REQUEST_TIMEOUT` must be a finite positive number (got {raw_env!r})."
            ) from exc
        if not math.isfinite(env_val) or env_val <= 0:
            raise ValueError(
                f"Inherited `COLAB_CLI_REQUEST_TIMEOUT` must be finite and > 0 (got {raw_env!r})."
            )
        return env_val, False
    return DEFAULT_REQUEST_TIMEOUT_S, False


class _BoundedStreamCapture:
    """Stream subprocess output to an optional private file while bounding RAM to O(1)."""

    _BEGIN_BYTES = PROBE_JSON_BEGIN.encode("utf-8")
    _END_BYTES = PROBE_JSON_END.encode("utf-8")
    _READY_BYTES = b"[colab] session ready"

    def __init__(
        self,
        *,
        log_path: Optional[Path] = None,
        max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
        capture_probe: bool = False,
        track_session_ready: bool = False,
        track_uncertain_alloc: bool = False,
    ) -> None:
        cap = max(int(max_capture_bytes), 512)
        self.max_head_bytes = cap // 2
        self.max_tail_bytes = cap - self.max_head_bytes
        self.log_path = log_path
        self.capture_probe = capture_probe
        self.track_session_ready = track_session_ready
        self.track_uncertain_alloc = track_uncertain_alloc

        self.total_bytes = 0
        self._head = bytearray()
        self._tail_chunks: deque[bytes] = deque()
        self._tail_len = 0

        self._probe_active = False
        self._probe_done = False
        self._probe_overlap = bytearray()
        self._probe_buf = bytearray()
        self._captured_probe: Optional[bytes] = None

        self._ready_overlap = bytearray()
        self.saw_session_ready = False
        self._uncertain_line_buf = bytearray()
        self.saw_pre_ready_uncertain = False

        self._fh: Optional[Any] = None
        self._tmp_path: Optional[Path] = None
        self.log_written = False
        self.write_error: Optional[str] = None
        self.untrusted_error: Optional[str] = None

    def _open_spool_if_needed(self) -> None:
        if self._fh is not None or self.log_path is None or self.write_error or self.untrusted_error:
            return
        try:
            if self.log_path.is_symlink() or (
                os.path.lexists(self.log_path) and not self.log_path.is_file()
            ):
                self.untrusted_error = f"Refusing unsafe log target path: {self.log_path}"
                return
            canonical_parent = _ensure_dir_not_shared_writable(self.log_path.parent)
            target = canonical_parent / self.log_path.name
            if target.is_symlink() or (os.path.lexists(target) and not target.is_file()):
                self.untrusted_error = f"Refusing unsafe log target path: {target}"
                return
            self.log_path = target
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(canonical_parent),
            )
            tmp_path = Path(tmp_name)
            try:
                os.chmod(tmp_path, 0o600)
                fh = os.fdopen(fd, "wb")
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            self._tmp_path = tmp_path
            self._fh = fh
        except UntrustedPathError as exc:
            self.untrusted_error = str(exc)
        except OSError as exc:
            self.write_error = f"Failed opening local run log {self.log_path}: {exc}"

    def _close_and_unlink_tmp(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self._tmp_path is not None:
            try:
                self._tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._tmp_path = None

    def _scan_probe(self, chunk: bytes) -> None:
        if not self.capture_probe or self._probe_done:
            return
        if not self._probe_active:
            combined = bytes(self._probe_overlap) + chunk
            idx = combined.find(self._BEGIN_BYTES)
            if idx != -1:
                self._probe_active = True
                self._probe_overlap.clear()
                tail_after_begin = combined[idx:]
                self._append_probe_bytes(tail_after_begin)
            else:
                keep = len(self._BEGIN_BYTES) - 1
                self._probe_overlap = bytearray(combined[-keep:]) if keep > 0 else bytearray()
        else:
            self._append_probe_bytes(chunk)

    def _append_probe_bytes(self, data: bytes) -> None:
        self._probe_buf.extend(data)
        end_idx = self._probe_buf.find(self._END_BYTES)
        if end_idx != -1:
            self._captured_probe = bytes(self._probe_buf[: end_idx + len(self._END_BYTES)])
            self._probe_buf.clear()
            self._probe_active = False
            self._probe_done = True
        elif len(self._probe_buf) > DEFAULT_MAX_PROBE_CAPTURE_BYTES:
            # Guard against unbounded memory if a script prints BEGIN without END.
            self._probe_buf.clear()
            self._probe_active = False
            self._probe_done = True

    def _scan_ready_and_uncertain(self, chunk: bytes) -> None:
        if not self.track_session_ready and not self.track_uncertain_alloc:
            return
        if self.track_uncertain_alloc and not self.saw_session_ready and not self.saw_pre_ready_uncertain:
            self._uncertain_line_buf.extend(chunk)
            while b"\n" in self._uncertain_line_buf:
                nl_idx = self._uncertain_line_buf.index(b"\n")
                raw_line = bytes(self._uncertain_line_buf[:nl_idx])
                del self._uncertain_line_buf[: nl_idx + 1]
                line_text = raw_line.decode("utf-8", errors="replace")
                if "[colab] session ready" in line_text.lower():
                    self.saw_session_ready = True
                    self._uncertain_line_buf.clear()
                    break
                if _UNCERTAIN_ALLOC_TRANSPORT_RE.search(line_text) or _UNCERTAIN_ALLOC_HTTP_RE.search(
                    line_text
                ):
                    self.saw_pre_ready_uncertain = True
                    self._uncertain_line_buf.clear()
                    break
            if len(self._uncertain_line_buf) > 4096:
                partial_text = bytes(self._uncertain_line_buf).decode("utf-8", errors="replace")
                if "[colab] session ready" in partial_text.lower():
                    self.saw_session_ready = True
                    self._uncertain_line_buf.clear()
                elif _UNCERTAIN_ALLOC_TRANSPORT_RE.search(
                    partial_text
                ) or _UNCERTAIN_ALLOC_HTTP_RE.search(partial_text):
                    self.saw_pre_ready_uncertain = True
                    self._uncertain_line_buf.clear()
                else:
                    del self._uncertain_line_buf[:-256]
        if self.track_session_ready and not self.saw_session_ready:
            combined = bytes(self._ready_overlap) + chunk.lower()
            if self._READY_BYTES in combined:
                self.saw_session_ready = True
                self._ready_overlap.clear()
                self._uncertain_line_buf.clear()
            else:
                keep = len(self._READY_BYTES) - 1
                self._ready_overlap = bytearray(combined[-keep:]) if keep > 0 else bytearray()

    def _flush_uncertain_line(self) -> None:
        if (
            self.track_uncertain_alloc
            and not self.saw_session_ready
            and not self.saw_pre_ready_uncertain
            and self._uncertain_line_buf
        ):
            line_text = bytes(self._uncertain_line_buf).decode("utf-8", errors="replace")
            if "[colab] session ready" in line_text.lower():
                self.saw_session_ready = True
            elif _UNCERTAIN_ALLOC_TRANSPORT_RE.search(line_text) or _UNCERTAIN_ALLOC_HTTP_RE.search(
                line_text
            ):
                self.saw_pre_ready_uncertain = True
            self._uncertain_line_buf.clear()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)

        if self.log_path is not None and self.write_error is None and self.untrusted_error is None:
            self._open_spool_if_needed()
            if self._fh is not None:
                try:
                    self._fh.write(chunk)
                except OSError as exc:
                    self.write_error = f"Failed writing local run log {self.log_path}: {exc}"
                    self._close_and_unlink_tmp()

        if len(self._head) < self.max_head_bytes:
            need = self.max_head_bytes - len(self._head)
            self._head.extend(chunk[:need])

        if self.max_tail_bytes > 0:
            if len(chunk) >= self.max_tail_bytes:
                self._tail_chunks.clear()
                self._tail_chunks.append(chunk[-self.max_tail_bytes :])
                self._tail_len = self.max_tail_bytes
            else:
                self._tail_chunks.append(chunk)
                self._tail_len += len(chunk)
                while self._tail_len > self.max_tail_bytes and self._tail_chunks:
                    overflow = self._tail_len - self.max_tail_bytes
                    first = self._tail_chunks[0]
                    if len(first) <= overflow:
                        self._tail_chunks.popleft()
                        self._tail_len -= len(first)
                    else:
                        self._tail_chunks[0] = first[overflow:]
                        self._tail_len -= overflow

        self._scan_probe(chunk)
        self._scan_ready_and_uncertain(chunk)

    def finalize(self, *, keep_log: bool, extra_suffix: bytes = b"") -> None:
        self._flush_uncertain_line()
        if not keep_log or self.log_path is None or self.write_error or self.untrusted_error:
            self._close_and_unlink_tmp()
            return
        if extra_suffix:
            self._open_spool_if_needed()
            if self._fh is not None:
                try:
                    self._fh.write(extra_suffix)
                except OSError as exc:
                    self.write_error = f"Failed writing local run log {self.log_path}: {exc}"
                    self._close_and_unlink_tmp()
                    return
        if self._fh is None or self._tmp_path is None:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
            if self.log_path.is_symlink() or (
                os.path.lexists(self.log_path) and not self.log_path.is_file()
            ):
                self.untrusted_error = f"Refusing unsafe log target path: {self.log_path}"
                self._close_and_unlink_tmp()
                return
            os.replace(self._tmp_path, self.log_path)
            self._tmp_path = None
            self.log_written = True
        except OSError as exc:
            self.write_error = f"Failed finalizing local run log {self.log_path}: {exc}"
            self._close_and_unlink_tmp()

    def captured_probe_text(self) -> Optional[str]:
        """Return the separately captured complete probe block, if observed in the stream."""
        if self._captured_probe is None:
            return None
        return self._captured_probe.decode("utf-8", errors="replace")

    def render_text(self) -> str:
        if self.total_bytes == 0:
            return ""
        tail_bytes = b"".join(self._tail_chunks)
        if self.total_bytes <= len(self._head):
            return bytes(self._head).decode("utf-8", errors="replace")
        if self.total_bytes <= len(self._head) + len(tail_bytes):
            overlap = (len(self._head) + len(tail_bytes)) - self.total_bytes
            combined = bytes(self._head) + tail_bytes[overlap:]
            return combined.decode("utf-8", errors="replace")

        omitted = self.total_bytes - len(self._head) - len(tail_bytes)
        parts: List[str] = [bytes(self._head).decode("utf-8", errors="replace")]
        parts.append(
            f"\n[colab_tpu_bench] ... {omitted} bytes omitted from bounded in-memory capture ...\n"
        )
        if self.saw_session_ready and self._READY_BYTES not in self._head.lower():
            parts.append("[colab] Session READY (observed in streamed stderr)\n")
        if self._captured_probe is not None:
            parts.append(self._captured_probe.decode("utf-8", errors="replace") + "\n")
        parts.append(tail_bytes.decode("utf-8", errors="replace"))
        return "".join(parts)


def _extract_run_session_from_cmd(cmd_list: Sequence[str]) -> Optional[str]:
    """Return the validated `-s <sess>` name from a `colab run` invocation, if present."""
    try:
        run_idx = cmd_list.index("run")
    except ValueError:
        return None
    for idx in range(run_idx + 1, len(cmd_list)):
        tok = cmd_list[idx]
        if tok == "--":
            break
        if tok in ("-s", "--session") and idx + 1 < len(cmd_list):
            candidate = cmd_list[idx + 1]
            if (
                candidate
                and ".." not in candidate
                and _SAFE_STORED_SESSION_RE.fullmatch(candidate)
            ):
                return candidate
            return None
    return None


def make_subprocess_runner(
    request_timeout_s: Optional[float] = None,
    run_wall_timeout_s: Optional[float] = None,
    log_dir: Optional[Path] = None,
    max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
) -> CommandRunner:
    """Create a non-interactive subprocess runner with bounded memory and optional log streaming."""

    def _run(cmd: Sequence[str]) -> CommandResult:
        cmd_list = list(cmd)
        env = os.environ.copy()
        effective_req_timeout, explicit_req = resolve_effective_request_timeout(request_timeout_s)
        if explicit_req or not (env.get("COLAB_CLI_REQUEST_TIMEOUT") or "").strip():
            env["COLAB_CLI_REQUEST_TIMEOUT"] = str(effective_req_timeout)
        # Preserve Colab CLI's same-request `GET /tun/m/assign?nbh=...` reconciliation on a
        # timed-out POST (default to 1 retry if unset; never force 0 so a lost response can
        # still be matched to its request ID). Fallback remains blocked on any uncertain error.
        if not (env.get("COLAB_CLI_ASSIGN_RETRIES") or "").strip():
            env["COLAB_CLI_ASSIGN_RETRIES"] = "1"
        run_session = _extract_run_session_from_cmd(cmd_list)
        is_run_cmd = run_session is not None or "run" in cmd_list
        if is_run_cmd:
            wall_timeout: Optional[float] = (
                run_wall_timeout_s
                if run_wall_timeout_s is not None and run_wall_timeout_s > 0
                else None
            )
        else:
            wall_timeout = max(effective_req_timeout * 2.0, 120.0)

        out_log_path = (
            log_dir / f"{run_session}.stdout.txt"
            if (log_dir is not None and run_session is not None)
            else None
        )
        err_log_path = (
            log_dir / f"{run_session}.stderr.txt"
            if (log_dir is not None and run_session is not None)
            else None
        )
        out_cap = _BoundedStreamCapture(
            log_path=out_log_path,
            max_capture_bytes=max_capture_bytes,
            capture_probe=True,
            track_session_ready=False,
        )
        err_cap = _BoundedStreamCapture(
            log_path=err_log_path,
            max_capture_bytes=max_capture_bytes,
            capture_probe=False,
            track_session_ready=True,
            track_uncertain_alloc=is_run_cmd,
        )

        started_mono = time.monotonic()
        proc = subprocess.Popen(
            cmd_list,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        def _drain_pipe(stream: Any, cap: _BoundedStreamCapture) -> None:
            if stream is None:
                return
            read_fn = getattr(stream, "read1", stream.read)
            try:
                while True:
                    chunk = read_fn(4096)
                    if not chunk:
                        break
                    cap.feed(chunk)
            except OSError:
                pass
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        t_out = threading.Thread(
            target=_drain_pipe, args=(proc.stdout, out_cap), daemon=True
        )
        t_err = threading.Thread(
            target=_drain_pipe, args=(proc.stderr, err_cap), daemon=True
        )
        t_out.start()
        t_err.start()

        def _collect_output(
            join_timeout: float = 2.0,
            *,
            keep_out_log: bool = False,
            keep_err_log: bool = False,
            err_suffix: bytes = b"",
        ) -> Tuple[str, str]:
            t_out.join(timeout=join_timeout)
            t_err.join(timeout=join_timeout)
            out_cap.finalize(keep_log=keep_out_log)
            err_cap.finalize(keep_log=keep_err_log, extra_suffix=err_suffix)
            return out_cap.render_text(), err_cap.render_text()

        def _combined_log_errors() -> Tuple[Optional[str], Optional[str]]:
            untrusted = out_cap.untrusted_error or err_cap.untrusted_error
            write_err = out_cap.write_error or err_cap.write_error
            return write_err, untrusted

        def _terminate_and_reap() -> None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=2.0)
                    except (subprocess.TimeoutExpired, OSError):
                        pass

        try:
            proc.wait(timeout=wall_timeout)
            elapsed_s = time.monotonic() - started_mono
            rc = proc.returncode if proc.returncode is not None else 1
            out_txt, err_txt = _collect_output(
                join_timeout=5.0,
                keep_out_log=(out_cap.total_bytes > 0),
                keep_err_log=(rc != 0 and err_cap.total_bytes > 0),
            )
            write_err, untrusted_err = _combined_log_errors()
            return CommandResult(
                returncode=rc,
                stdout=out_txt,
                stderr=err_txt,
                timed_out=False,
                stdout_log_written=out_cap.log_written,
                stderr_log_written=err_cap.log_written,
                log_write_error=write_err,
                untrusted_log_error=untrusted_err,
                captured_probe_text=out_cap.captured_probe_text(),
                elapsed_s=elapsed_s,
                pre_ready_uncertain=err_cap.saw_pre_ready_uncertain,
            )
        except subprocess.TimeoutExpired:
            _terminate_and_reap()
            elapsed_s = time.monotonic() - started_mono
            timeout_label = f"{wall_timeout:.1f}s" if wall_timeout is not None else "configured"
            note = (
                f"[colab_tpu_bench] `colab` subprocess exceeded wall timeout "
                f"({timeout_label}); stopping session and reconciling active sessions."
            )
            suffix_bytes = ("\n" + note + "\n").encode("utf-8") if err_cap.total_bytes > 0 else (note + "\n").encode("utf-8")
            out_txt, err_txt = _collect_output(
                join_timeout=2.0,
                keep_out_log=(out_cap.total_bytes > 0),
                keep_err_log=True,
                err_suffix=suffix_bytes,
            )
            combined_err = f"{err_txt}\n{note}".strip() if err_txt else note
            write_err, untrusted_err = _combined_log_errors()
            return CommandResult(
                returncode=124,
                stdout=out_txt,
                stderr=combined_err + "\n",
                timed_out=True,
                stdout_log_written=out_cap.log_written,
                stderr_log_written=err_cap.log_written,
                log_write_error=write_err,
                untrusted_log_error=untrusted_err,
                captured_probe_text=out_cap.captured_probe_text(),
                elapsed_s=elapsed_s,
                pre_ready_uncertain=err_cap.saw_pre_ready_uncertain,
            )
        except (KeyboardInterrupt, TerminationSignal) as exc:
            _terminate_and_reap()
            elapsed_s = time.monotonic() - started_mono
            is_sigterm = isinstance(exc, TerminationSignal)
            cancel_note = (
                "[colab_tpu_bench] Terminated by supervisor (SIGTERM)."
                if is_sigterm
                else "[colab_tpu_bench] Interrupted by operator (KeyboardInterrupt)."
            )
            suffix_bytes = (
                ("\n" + cancel_note + "\n").encode("utf-8")
                if err_cap.total_bytes > 0
                else (cancel_note + "\n").encode("utf-8")
            )
            out_txt, err_txt = _collect_output(
                join_timeout=2.0,
                keep_out_log=(out_cap.total_bytes > 0),
                keep_err_log=True,
                err_suffix=suffix_bytes,
            )
            write_err, untrusted_err = _combined_log_errors()
            raise CommandCancelledError(
                signal_name="SIGTERM" if is_sigterm else "SIGINT",
                returncode=143 if is_sigterm else 130,
                stdout=out_txt,
                stderr=err_txt,
                stdout_log_written=out_cap.log_written,
                stderr_log_written=err_cap.log_written,
                log_write_error=write_err,
                untrusted_log_error=untrusted_err,
                captured_probe_text=out_cap.captured_probe_text(),
                elapsed_s=elapsed_s,
            ) from exc

    return _run


def default_command_runner(cmd: Sequence[str]) -> CommandResult:
    """Execute a subprocess command and capture stdout/stderr as UTF-8."""
    return make_subprocess_runner()(cmd)


def parse_usage_output(text: str) -> UsageSnapshot:
    """Parse `colab usage` output into structured signed CU accounting fields."""
    balance_match = re.search(
        r"Current balance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s+compute units",
        text,
        re.IGNORECASE,
    )
    rate_match = re.search(
        r"Usage rate:\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*/\s*hr",
        text,
        re.IGNORECASE,
    )
    active_match = re.search(
        r"Active assignments:\s*([+-]?[0-9]+)",
        text,
        re.IGNORECASE,
    )
    return UsageSnapshot(
        balance_cu=float(balance_match.group(1)) if balance_match else None,
        usage_rate_per_hour=float(rate_match.group(1)) if rate_match else None,
        active_assignments=int(active_match.group(1)) if active_match else None,
        raw_output=text.strip(),
    )


def parse_sessions_output(text: str) -> Tuple[Set[str], Set[str]]:
    """Parse `colab sessions` output into `(active_session_names, active_endpoints)`."""
    names: Set[str] = set()
    endpoints: Set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SESSION_LINE_RE.match(line)
        if not m:
            continue
        sess_name, endpoint = m.group(1), m.group(2)
        if sess_name and sess_name != "?":
            names.add(sess_name)
        endpoints.add(endpoint)
    return names, endpoints


def inspect_exported_notebook(nb_path: Path) -> Tuple[Optional[str], bool]:
    """Return `(endpoint_id, has_workload_code_cell)` from a `colab log -o *.ipynb` export."""
    if nb_path.is_symlink() or not nb_path.is_file():
        return None, False
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False
    cells = data.get("cells")
    if not isinstance(cells, list):
        return None, False
    endpoint: Optional[str] = None
    has_code_cell = False
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(str(part) for part in src)
        src_str = str(src)
        if cell_type == "markdown" and endpoint is None:
            m = re.search(r"-\s*Endpoint:\s*`?([^`\s\n]+)`?", src_str)
            if m:
                endpoint = m.group(1)
        elif cell_type == "code" and src_str.strip():
            has_code_cell = True
    return endpoint, has_code_cell


def extract_endpoint_from_notebook(nb_path: Path) -> Optional[str]:
    """Extract server endpoint ID recorded in a `colab log -o *.ipynb` session header."""
    endpoint, _ = inspect_exported_notebook(nb_path)
    return endpoint


def read_endpoint_from_history(
    session_name: str,
    history_dir: Optional[Path] = None,
) -> Optional[str]:
    """Read the last `session_created` endpoint for `session_name` from Colab CLI history JSONL.

    Colab CLI 0.8.0 records `session_created` (including `endpoint`) in
    `~/.config/colab-cli/history/<session_name>.jsonl` immediately on allocation, even if
    `colab run` later removes the local session mapping or `colab log -o *.ipynb` fails
    (e.g., when the artifact volume is full). Never returns or logs non-endpoint fields.
    """
    if (
        not isinstance(session_name, str)
        or not session_name
        or ".." in session_name
        or not _SAFE_STORED_SESSION_RE.fullmatch(session_name)
    ):
        return None
    raw_dir = (
        Path(os.path.expanduser(str(history_dir)))
        if history_dir is not None
        else Path(os.path.expanduser("~/.config/colab-cli/history"))
    )
    if not os.path.lexists(raw_dir) or os.path.islink(raw_dir):
        return None
    try:
        dir_st = os.lstat(raw_dir)
    except OSError:
        return None
    if not stat.S_ISDIR(dir_st.st_mode):
        return None
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    if euid is not None and dir_st.st_uid != euid:
        return None
    if (dir_st.st_mode & 0o022) != 0:
        return None

    log_path = raw_dir / f"{session_name}.jsonl"
    if not os.path.lexists(log_path) or os.path.islink(log_path):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(log_path, flags)
    except OSError:
        return None
    endpoint: Optional[str] = None
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            return None
        if euid is not None and fst.st_uid != euid:
            return None
        if (fst.st_mode & 0o022) != 0:
            return None
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
            fd = -1
            for raw_line in fh:
                if len(raw_line) > 256 * 1024:
                    continue
                line = raw_line.strip()
                if not line or '"session_created"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(obj, dict)
                    and obj.get("event_type") == "session_created"
                    and isinstance(obj.get("endpoint"), str)
                    and obj["endpoint"].strip()
                ):
                    endpoint = obj["endpoint"].strip()
    except OSError:
        return endpoint
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    return endpoint


def read_endpoint_from_session_store(
    config_path: Path,
    session_name: str,
) -> Optional[str]:
    """Read only the non-secret `endpoint` for `session_name` from an isolated session store."""
    if not os.path.lexists(config_path) or os.path.islink(config_path):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(config_path, flags)
    except OSError:
        return None
    try:
        fst = os.fstat(fd)
        euid = os.geteuid() if hasattr(os, "geteuid") else None
        if (
            not stat.S_ISREG(fst.st_mode)
            or (euid is not None and fst.st_uid != euid)
            or (fst.st_mode & 0o022) != 0
            or fst.st_size > 2 * 1024 * 1024
        ):
            return None
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
            fd = -1
            data = json.loads(fh.read())
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    if not isinstance(data, dict):
        return None
    entry = data.get(session_name)
    if isinstance(entry, dict) and isinstance(entry.get("endpoint"), str) and entry["endpoint"].strip():
        return entry["endpoint"].strip()
    return None


def detect_cli_capabilities(
    colab_bin: str, runner: CommandRunner
) -> Dict[str, bool]:
    """Inspect `colab --help` safely across CLI releases, including when absent."""
    try:
        res = runner([colab_bin, "--help"])
    except FileNotFoundError:
        return {
            "cli_found": False,
            "help_ok": False,
            "has_public_api": False,
            "has_no_public_api": False,
            "has_logtostderr": True,
            "has_config": True,
            "has_auth": True,
        }
    if res.returncode == 127:
        return {
            "cli_found": False,
            "help_ok": False,
            "has_public_api": False,
            "has_no_public_api": False,
            "has_logtostderr": True,
            "has_config": True,
            "has_auth": True,
        }
    combined = f"{res.stdout}\n{res.stderr}"
    help_ok = res.returncode == 0 and bool(combined.strip())
    return {
        "cli_found": True,
        "help_ok": help_ok,
        "has_public_api": bool(help_ok and re.search(r"(?<![A-Za-z0-9-])--public-api\b", combined)),
        "has_no_public_api": bool(
            help_ok and re.search(r"(?<![A-Za-z0-9-])--no-public-api\b", combined)
        ),
        "has_logtostderr": (
            True if not help_ok else bool(re.search(r"(?<![A-Za-z0-9-])--logtostderr\b", combined))
        ),
        # Never drop required `--config` or `--auth` flags if `--help` failed or was truncated.
        "has_config": True,
        "has_auth": True,
    }


def build_base_cmd(
    cfg: BenchConfig,
    caps: Dict[str, bool],
    isolated_config_path: Path,
) -> List[str]:
    """Build the global `colab` prefix before the subcommand.

    Always passes explicit `--auth` and `--config` for supported `google-colab-cli >= 0.7.6`
    releases so a failed or truncated `colab --help` can never silently fall back to CLI
    defaults (cached OAuth2 account or the user's shared `~/.config/colab-cli/sessions.json`).
    """
    if not cfg.use_no_public_api:
        raise ValueError(
            "`google-colab-cli` still routes `usage`, `sessions`, `run` allocation, and "
            "teardown through the `/tun/m/*` tunnel even when `--public-api` is advertised; "
            "omit `--public-api` here or use Official Colab MCP / Public REST `v1beta` "
            "for Public-API-only environments."
        )
    cmd = [cfg.colab_bin]
    if cfg.auth:
        cmd.extend(["--auth", cfg.auth])
    cmd.extend(["--config", str(isolated_config_path)])
    if caps.get("has_no_public_api", False):
        cmd.append("--no-public-api")
    return cmd


def is_capacity_or_quota_error(
    returncode: int, stdout: str, stderr: str
) -> bool:
    """Return True iff `colab run` failed with a confirmed pre-allocation refusal."""
    del stdout  # Remote workload stdout must never trigger TPU re-execution.
    if returncode == 0:
        return False
    err_lower = stderr.lower()
    # Once `colab run` prints `[colab] Session READY (...)`, the VM was allocated
    # and the user workload began executing; never fall back and duplicate side effects.
    if "[colab] session ready" in err_lower:
        return False
    # Never treat a run that experienced a timeout, 5xx, transport disconnect, or internal
    # allocation retry as a clean capacity refusal even if a subsequent retry returned 412.
    if is_allocation_timeout_error(returncode, stderr):
        return False
    in_rich_box = False
    in_std_tb = False
    for raw_line in stderr.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if stripped.startswith("╭"):
            in_rich_box = True
            continue
        if in_rich_box:
            if stripped.startswith("╰"):
                in_rich_box = False
            continue
        if lowered.startswith("traceback (most recent call last):"):
            in_std_tb = True
            continue
        if in_std_tb:
            if raw_line[:1].isspace() or lowered.startswith('file "'):
                continue
            in_std_tb = False
        if _TRACEBACK_CODE_LINE_RE.match(lowered):
            continue
        if any(pat.search(lowered) for pat in _ALLOCATION_DIAGNOSTIC_PATTERNS):
            return True
    return False


def is_wall_timeout_error(res: CommandResult) -> bool:
    """Return True iff the local `colab` subprocess hit its wall-clock deadline."""
    return bool(res.timed_out)


_UNCERTAIN_ALLOC_HTTP_RE = re.compile(
    r"(?:\bhttp(?:\s+error)?\s*5\d\d\b|"
    r"\bresponse:\s*5\d\d\b|"
    r"\b5\d\d\s+(?:server error|internal server error|bad gateway|service unavailable|gateway time-?out)\b|"
    r"\bstatus(?:[\s_]+code)?\s*[=:]\s*5\d\d\b|"
    r"\b(?:internal server error|bad gateway|service unavailable|gateway time-?out)\b)",
    re.IGNORECASE,
)

_UNCERTAIN_ALLOC_TRANSPORT_RE = re.compile(
    r"(?:\breadtimeout\b|\bread timed out\b|\bconnecttimeout\b|\bconnect timed out\b|"
    r"\bassignment request failed;\s*retrying\b|"
    r"\bconnectionerror\b|\bremotedisconnected\b|\bremote end closed connection\b|"
    r"\bconnectionreseterror\b|\bconnection reset\b|\bconnectionabortederror\b|\bconnection aborted\b|"
    r"\bbrokenpipeerror\b|\bbroken pipe\b|\bchunkedencodingerror\b|\bprotocolerror\b|"
    r"\bsslerror\b|\bssleoferror\b|\bunexpected eof\b|\bproxyerror\b|\brequestexception\b|"
    r"\bmaxretryerror\b|\bmax retries exceeded\b|\bnewconnectionerror\b|"
    r"\bincompleteread\b|\bsocket\.timeout\b|\btimeouterror\b)",
    re.IGNORECASE,
)


def is_allocation_timeout_error(
    returncode: int,
    stderr: str,
    *,
    timed_out: bool = False,
) -> bool:
    """Return True if allocation or setup ended with an uncertain server-side outcome."""
    if returncode == 0 and not timed_out:
        return False
    err_lower = stderr.lower()
    if timed_out:
        return "[colab] session ready" not in err_lower
    if "[colab] session ready" in err_lower:
        return False
    if _UNCERTAIN_ALLOC_TRANSPORT_RE.search(stderr):
        return True
    return bool(_UNCERTAIN_ALLOC_HTTP_RE.search(stderr))


def _contains_oauth_prompt_failure(res: CommandResult) -> bool:
    combined = f"{res.stdout}\n{res.stderr}".lower()
    return any(marker in combined for marker in _OAUTH_PROMPT_MARKERS)


def _has_colab_auth_failure(res: CommandResult) -> bool:
    """Return True if `colab` emitted an authentication/credential failure (even on exit 0)."""
    combined = f"{res.stdout}\n{res.stderr}".lower()
    return any(marker in combined for marker in _COLAB_AUTH_ERROR_MARKERS)


def extract_probe_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract JSON payload emitted between FreeTPU probe markers.

    Pairs each closing marker with its nearest preceding opening marker so a truncated
    opening marker in a bounded head buffer can never mask an intact payload that follows.
    """
    if not stdout:
        return None
    search_pos = 0
    while True:
        end_idx = stdout.find(PROBE_JSON_END, search_pos)
        if end_idx == -1:
            return None
        begin_idx = stdout.rfind(PROBE_JSON_BEGIN, 0, end_idx)
        if begin_idx != -1:
            candidate = stdout[begin_idx + len(PROBE_JSON_BEGIN) : end_idx].strip()
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                pass
        search_pos = end_idx + len(PROBE_JSON_END)


def validate_builtin_probe(probe_payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Validate built-in probe JSON proved TPU execution and a valid GEMM benchmark."""
    if not isinstance(probe_payload, dict):
        return "Missing built-in probe JSON output from remote run."
    if probe_payload.get("jax_error"):
        return f"Built-in probe reported JAX/TPU error: {probe_payload['jax_error']}"
    jax_info = probe_payload.get("jax")
    if not isinstance(jax_info, dict):
        return "Built-in probe output is missing `jax` runtime metadata."
    if jax_info.get("default_backend") != "tpu":
        return (
            f"Built-in probe did not run on TPU (jax.default_backend={jax_info.get('default_backend')!r})."
        )
    if (jax_info.get("device_count") or 0) <= 0:
        return "Built-in probe reported zero JAX TPU devices."
    devices = jax_info.get("devices")
    if not isinstance(devices, list) or not any(
        isinstance(d, dict) and d.get("platform") == "tpu" for d in devices
    ):
        return "Built-in probe reported no device with platform=='tpu'."
    bench = probe_payload.get("benchmark")
    if not isinstance(bench, dict):
        return "Built-in probe did not complete the BF16 GEMM benchmark."
    if bench.get("hardware") != "colab-tpu":
        return f"Unexpected benchmark hardware tag: {bench.get('hardware')!r}."
    hw_us = bench.get("hw_us")
    gflops = bench.get("gflops")
    if not isinstance(hw_us, (int, float)) or hw_us <= 0:
        return f"Invalid benchmark latency hw_us={hw_us!r}."
    if not isinstance(gflops, (int, float)) or gflops <= 0:
        return f"Invalid benchmark throughput gflops={gflops!r}."
    return None


def default_tpu_probe_script() -> str:
    """Return a clean-room remote Python script for Colab TPU probing."""
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import glob
        import json
        import os
        import platform
        import shutil
        import subprocess
        import sys
        import time

        def run_cmd(args):
            try:
                p = subprocess.run(args, capture_output=True, text=True, timeout=30)
                return {{"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}}
            except Exception as exc:
                return {{"error": str(exc)}}

        report = {{
            "schema": "freetpu.colab_tpu_probe.v1",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "tpu_env": {{
                k: v for k, v in os.environ.items()
                if k.startswith(("TPU_", "PJRT_", "JAX_", "XLA_"))
            }},
            "vfio_devices": sorted(glob.glob("/dev/vfio/*")),
            "tpu_info": run_cmd(["tpu-info"]) if shutil.which("tpu-info") else {{"available": False}},
            "jax": None,
            "benchmark": None,
        }}

        exit_rc = 1
        try:
            import jax
            import jax.numpy as jnp

            devices = jax.devices()
            default_backend = jax.default_backend()
            dev_info = []
            for d in devices:
                entry = {{
                    "id": getattr(d, "id", None),
                    "platform": getattr(d, "platform", None),
                    "device_kind": getattr(d, "device_kind", None),
                }}
                try:
                    entry["memory_stats"] = d.memory_stats()
                except Exception:
                    pass
                dev_info.append(entry)

            report["jax"] = {{
                "version": getattr(jax, "__version__", None),
                "default_backend": default_backend,
                "device_count": len(devices),
                "devices": dev_info,
            }}

            tpu_devices = [d for d in devices if getattr(d, "platform", None) == "tpu"]
            if default_backend != "tpu" or not tpu_devices:
                raise RuntimeError(
                    f"Expected JAX TPU backend, got default_backend={{default_backend!r}} with {{len(tpu_devices)}} TPU device(s)"
                )

            # Canonical small BF16 GEMM microbenchmark (Track A/B parity schema).
            m, k, n = 64, 512, 512
            key = jax.random.PRNGKey(0)
            a = jax.random.normal(key, (m, k), dtype=jnp.bfloat16)
            b = jax.random.normal(key, (k, n), dtype=jnp.bfloat16)

            @jax.jit
            def matmul_fn(x, y):
                return jnp.matmul(x, y)

            # Warmup compilation + device transfer.
            out = matmul_fn(a, b)
            out.block_until_ready()

            iters = 20
            t0 = time.perf_counter()
            for _ in range(iters):
                out = matmul_fn(a, b)
                out.block_until_ready()
            dt_us = ((time.perf_counter() - t0) / iters) * 1e6
            flops = 2.0 * m * k * n
            gflops = (flops / (dt_us * 1e-6)) / 1e9

            report["benchmark"] = {{
                "hardware": "colab-tpu",
                "silicon": tpu_devices[0].device_kind,
                "dtype": "bf16",
                "m": m,
                "k": k,
                "n": n,
                "hw_us": round(dt_us, 3),
                "gflops": round(gflops, 3),
                "stack": f"jax-{{getattr(jax, '__version__', 'unknown')}}",
            }}
            exit_rc = 0
        except Exception as exc:
            report["jax_error"] = str(exc)
            exit_rc = 1
        finally:
            print("{PROBE_JSON_BEGIN}")
            print(json.dumps(report, indent=2, sort_keys=True))
            print("{PROBE_JSON_END}")

        if exit_rc != 0:
            sys.exit(exit_rc)
        """
    )


def _interactive_oauth2_preflight(base_cmd: Sequence[str]) -> Optional[str]:
    """Ensure `--auth oauth2` has valid cached credentials interactively on TTY before capture."""
    whoami_cmd = [*base_cmd, "whoami"]
    try:
        proc = subprocess.run(whoami_cmd, stdout=sys.stderr, check=False)
    except OSError as exc:
        return f"Failed to run interactive `{base_cmd[0]} whoami`: {exc}"
    if proc.returncode != 0:
        return (
            "OAuth2 credential check (`colab --auth oauth2 whoami`) failed; complete "
            "authorization interactively or use `--auth adc`."
        )
    return None


def run_benchmark(
    cfg: BenchConfig,
    runner: Optional[CommandRunner] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Execute pre-flight checks, TPU run (with conservative fallback), and release verification."""
    if not math.isfinite(cfg.min_cu_balance):
        return 2, {
            "status": "invalid_cu_balance",
            "error": f"`--min-cu-balance` must be a finite number (got {cfg.min_cu_balance!r}).",
        }
    if not math.isfinite(cfg.timeout_s) or cfg.timeout_s < 0:
        return 2, {
            "status": "invalid_timeout",
            "error": f"`--timeout` must be finite and >= 0 (got {cfg.timeout_s!r}).",
        }
    if cfg.wall_timeout_s is not None and (
        not math.isfinite(cfg.wall_timeout_s) or cfg.wall_timeout_s < 0
    ):
        return 2, {
            "status": "invalid_timeout",
            "error": f"`--wall-timeout` must be finite and >= 0 (got {cfg.wall_timeout_s!r}).",
        }
    try:
        effective_request_timeout, _ = resolve_effective_request_timeout(cfg.request_timeout_s)
    except ValueError as exc:
        return 2, {
            "status": "invalid_timeout",
            "error": str(exc),
        }
    if not math.isfinite(cfg.teardown_retry_delay_s) or cfg.teardown_retry_delay_s < 0:
        return 2, {
            "status": "invalid_timeout",
            "error": (
                f"`teardown_retry_delay_s` must be finite and >= 0 "
                f"(got {cfg.teardown_retry_delay_s!r})."
            ),
        }

    effective_wall_timeout = resolve_effective_wall_timeout(cfg.timeout_s, cfg.wall_timeout_s)
    using_default_runner = runner is None

    for arg in cfg.script_args:
        if arg == "--env" or arg.startswith("--env="):
            return 2, {
                "status": "unsupported_script_arg",
                "error": (
                    "Forwarding `--env` via script arguments is blocked because `colab run` "
                    "parses `--env` even after `--`."
                ),
            }

    if cfg.session_name is not None and (
        not _SAFE_SESSION_RE.fullmatch(cfg.session_name) or ".." in cfg.session_name
    ):
        return 2, {
            "status": "invalid_session_name",
            "error": (
                f"Session prefix {cfg.session_name!r} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}} "
                "and must not contain '..'."
            ),
        }

    try:
        if cfg.out_dir is None:
            env_tmp = os.environ.get("TMPDIR")
            tmp_root = Path(env_tmp if env_tmp else tempfile.gettempdir())
            canonical_tmp_root = _resolve_and_validate_existing_dir(tmp_root)
            created_tmp = Path(
                tempfile.mkdtemp(prefix="freetpu-colab-", dir=str(canonical_tmp_root))
            )
            try:
                out_dir = _ensure_private_dir(created_tmp)
            except Exception:
                shutil.rmtree(created_tmp, ignore_errors=True)
                raise
        else:
            out_dir = _ensure_private_dir(cfg.out_dir)
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_out_dir", "error": str(exc)}

    if runner is None:
        runner = make_subprocess_runner(
            cfg.request_timeout_s,
            run_wall_timeout_s=effective_wall_timeout,
            log_dir=out_dir,
        )

    session_prefix = cfg.session_name or "freetpu"
    session_base = f"{session_prefix}-{uuid.uuid4().hex[:8]}"

    try:
        isolated_config = _normalize_config_path(cfg.config_path, out_dir, session_base)
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_config_path", "error": str(exc)}

    collision_err = _check_config_artifact_collision(
        isolated_config, out_dir, session_base, cfg.tpus, script_path=cfg.script_path
    )
    if collision_err is not None:
        return 2, {"status": "config_artifact_collision", "error": collision_err}

    try:
        has_existing_store_entries = _validate_existing_session_store(isolated_config)
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_config_path", "error": str(exc)}

    script_file = cfg.script_path
    generated_script = False
    wrapped_custom_script = False
    if script_file is None:
        script_file = out_dir / "colab_tpu_probe_payload.py"
        try:
            _write_text_nofollow(script_file, default_tpu_probe_script())
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        generated_script = True
    elif script_file.is_symlink() or not script_file.is_file():
        return 2, {"status": "invalid_script", "error": f"Invalid script path: {script_file}"}
    else:
        try:
            script_file, wrapped_custom_script = _prepare_custom_script(
                script_file, out_dir, session_base
            )
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        except ValueError as exc:
            return 2, {"status": "invalid_script", "error": str(exc)}

    caps = detect_cli_capabilities(cfg.colab_bin, runner)
    if not cfg.dry_run and not caps.get("cli_found", True):
        return 2, {
            "status": "missing_colab_cli",
            "error": (
                f"Executable {cfg.colab_bin!r} not found in PATH. Install via `uv tool install google-colab-cli`."
            ),
        }

    listing_config = out_dir / f"colab-sessions-list.{session_base}.json"
    try:
        base_cmd = build_base_cmd(cfg, caps, isolated_config)
        list_base_cmd = build_base_cmd(cfg, caps, listing_config)
    except ValueError as exc:
        return 2, {"status": "unsupported_public_api_mode", "error": str(exc)}

    has_logtostderr = bool(caps.get("has_logtostderr", True))
    run_base_cmd = [*base_cmd, "--logtostderr"] if has_logtostderr else list(base_cmd)
    retry_signal_observable = has_logtostderr or (
        (os.environ.get("COLAB_CLI_ASSIGN_RETRIES") or "").strip() == "0"
    )

    planned_commands: List[List[str]] = [
        [*base_cmd, "usage"],
        [*list_base_cmd, "sessions"],
    ]
    for idx, tpu in enumerate(cfg.tpus):
        sess = f"{session_base}-{tpu}" if idx > 0 else session_base
        run_cmd = [
            *run_base_cmd,
            "run",
            "-s",
            sess,
            "--tpu",
            tpu.upper(),
            "--timeout",
            str(cfg.timeout_s),
            "--",
            str(script_file),
            *cfg.script_args,
        ]
        log_cmd = [*base_cmd, "log", "-s", sess, "-o", str(out_dir / f"{sess}.ipynb")]
        planned_commands.append(run_cmd)
        planned_commands.append(log_cmd)
    planned_commands.append([*list_base_cmd, "sessions"])

    if cfg.dry_run:
        plan_path = out_dir / "dry-run-plan.json"
        summary = {
            "status": "dry_run",
            "colab_cli_found": bool(caps.get("cli_found", True)),
            "out_dir": str(out_dir),
            "plan_path": str(plan_path),
            "isolated_config": str(isolated_config),
            "script": str(script_file),
            "generated_script": generated_script,
            "wrapped_custom_script": wrapped_custom_script,
            "session_base": session_base,
            "tpus": list(cfg.tpus),
            "timeout_s": cfg.timeout_s,
            "wall_timeout_s": cfg.wall_timeout_s,
            "effective_wall_timeout_s": effective_wall_timeout,
            "request_timeout_s": cfg.request_timeout_s,
            "effective_request_timeout_s": effective_request_timeout,
            "planned_commands": planned_commands,
        }
        try:
            _write_text_nofollow(
                plan_path,
                json.dumps(summary, indent=2) + "\n",
            )
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        return 0, summary

    ver_ok, cli_version, ver_err, ver_rc = check_colab_cli_version(cfg.colab_bin, runner)
    if not ver_ok:
        return 2, {
            "status": "unsupported_colab_cli_version",
            "detected_version": cli_version,
            "min_version": ".".join(str(x) for x in MIN_COLAB_CLI_VERSION),
            "returncode": ver_rc,
            "error": ver_err,
        }

    if cfg.auth == "oauth2" and using_default_runner:
        if sys.stdin.isatty():
            oauth_err = _interactive_oauth2_preflight(base_cmd)
            if oauth_err is not None:
                return 1, {"status": "oauth2_auth_required", "error": oauth_err}

    # 1. Pre-flight Compute Unit budget & baseline active session snapshot.
    usage_res = runner([*base_cmd, "usage"])
    if usage_res.returncode != 0:
        if cfg.auth == "oauth2" and _contains_oauth_prompt_failure(usage_res):
            return 1, {
                "status": "oauth2_auth_required",
                "returncode": usage_res.returncode,
                "error": (
                    "OAuth2 token is not cached and cannot prompt in non-interactive capture mode; "
                    "run `colab --auth=oauth2 whoami` in an interactive terminal or pass `--auth adc`."
                ),
                "stderr": usage_res.stderr.strip(),
            }
        return 1, {
            "status": "usage_check_failed",
            "returncode": usage_res.returncode,
            "stderr": usage_res.stderr.strip(),
            "stdout": usage_res.stdout.strip(),
        }

    usage = parse_usage_output(usage_res.stdout)
    if usage.balance_cu is None:
        return 1, {
            "status": "usage_parse_failed",
            "usage": asdict(usage),
            "error": "Could not parse signed Compute Unit balance from `colab usage` output.",
        }
    if cfg.max_active_assignments is not None and usage.active_assignments is None:
        return 1, {
            "status": "usage_parse_failed",
            "usage": asdict(usage),
            "error": "Could not parse Active assignments count from `colab usage` output.",
        }
    if usage.balance_cu < cfg.min_cu_balance:
        return 1, {
            "status": "cu_budget_insufficient",
            "usage": asdict(usage),
            "min_cu_balance": cfg.min_cu_balance,
        }
    if (
        cfg.max_active_assignments is not None
        and usage.active_assignments is not None
        and usage.active_assignments >= cfg.max_active_assignments
    ):
        return 1, {
            "status": "active_assignments_limit_reached",
            "usage": asdict(usage),
            "max_active_assignments": cfg.max_active_assignments,
        }

    # Use a throwaway per-run empty config for `colab sessions` so `sync_sessions()` never
    # prunes existing mappings from an explicit `--config` (e.g., when credentials change).
    sess_before_res = runner([*list_base_cmd, "sessions"])
    for _tmp_list_file in (listing_config, Path(f"{listing_config}.lock")):
        try:
            _tmp_list_file.unlink(missing_ok=True)
        except OSError:
            pass
    if sess_before_res.returncode != 0 or _has_colab_auth_failure(sess_before_res):
        return 1, {
            "status": "sessions_check_failed",
            "returncode": sess_before_res.returncode,
            "stderr": sess_before_res.stderr.strip(),
            "stdout": sess_before_res.stdout.strip(),
        }
    names_before, endpoints_before = parse_sessions_output(sess_before_res.stdout)

    # 2. Ephemeral TPU run with conservative capacity fallback (v6e1 -> v5e1).
    attempts: List[Dict[str, Any]] = []
    attempted_sessions: List[str] = []
    final_rc = 1
    chosen_tpu: Optional[str] = None
    chosen_session: Optional[str] = None
    created_endpoint: Optional[str] = None
    probe_payload: Optional[Dict[str, Any]] = None
    proof_notebook: Optional[Path] = None
    metadata_notebook: Optional[Path] = None
    allocation_timed_out = False
    wall_timed_out = False
    interrupted = False
    terminated = False
    ready_seen_any = False
    maybe_allocated = False
    probe_error: Optional[str] = None
    untrusted_target_error: Optional[str] = None
    artifact_write_error: Optional[str] = None
    export_attempted = False
    export_ok: Optional[bool] = None
    export_metadata_only = False
    export_returncode: Optional[int] = None
    export_error: Optional[str] = None
    export_stderr_tail: Optional[str] = None
    forced_stop: Dict[str, Any] = {
        "attempted": False,
        "session": None,
        "returncode": None,
        "error": None,
        "stderr_tail": None,
    }
    release_retries: List[Dict[str, Any]] = []
    recovery_command: Optional[str] = None
    recovery_config_path: Optional[Path] = None
    in_cleanup = False
    cleanup_wake_event = threading.Event()

    def _record_cleanup_signal(exc: BaseException) -> Tuple[int, str]:
        nonlocal terminated, interrupted
        if isinstance(exc, CommandCancelledError):
            if exc.signal_name == "SIGTERM":
                terminated = True
                return exc.returncode, "SIGTERM"
            interrupted = True
            return exc.returncode, "SIGINT"
        if isinstance(exc, TerminationSignal):
            terminated = True
            return 143, "SIGTERM"
        interrupted = True
        return 130, "SIGINT"

    def _run_stop_cmd(
        sess_name: str,
        *,
        cmd_prefix: Optional[Sequence[str]] = None,
    ) -> Tuple[CommandResult, Optional[str]]:
        prefix = list(cmd_prefix) if cmd_prefix is not None else base_cmd
        try:
            stop_res = runner([*prefix, "stop", "-s", sess_name])
        except (TerminationSignal, CommandCancelledError) as exc:
            sig_rc, sig_name = _record_cleanup_signal(exc)
            msg = f"`colab stop -s {sess_name}` interrupted by {sig_name}"
            return CommandResult(sig_rc, "", msg + "\n"), msg
        err_msg: Optional[str] = None
        if stop_res.returncode != 0:
            diag = (
                stop_res.stderr.strip()
                or stop_res.stdout.strip()
                or "no diagnostic output"
            )
            err_msg = f"`colab stop -s {sess_name}` failed (rc={stop_res.returncode}): {diag}"
        elif _has_colab_auth_failure(stop_res):
            diag = stop_res.stderr.strip() or stop_res.stdout.strip()
            err_msg = f"`colab stop -s {sess_name}` exited 0 with authentication error: {diag}"
        return stop_res, err_msg

    def _recover_session_endpoint(sess_name: str) -> Optional[str]:
        nonlocal created_endpoint
        ep = read_endpoint_from_history(sess_name)
        if not ep:
            ep = read_endpoint_from_session_store(isolated_config, sess_name)
        if ep:
            created_endpoint = ep
        return ep

    def _export_session_log(sess_name: str) -> None:
        nonlocal export_attempted, export_ok, export_metadata_only
        nonlocal export_returncode, export_error, export_stderr_tail
        nonlocal proof_notebook, metadata_notebook, created_endpoint, untrusted_target_error
        nb_path = out_dir / f"{sess_name}.ipynb"
        if nb_path.is_symlink() or (os.path.lexists(nb_path) and not nb_path.is_file()):
            untrusted_target_error = f"Refusing unsafe notebook target path: {nb_path}"
            return
        export_attempted = True
        try:
            log_res = runner([*base_cmd, "log", "-s", sess_name, "-o", str(nb_path)])
        except (TerminationSignal, CommandCancelledError) as exc:
            sig_rc, sig_name = _record_cleanup_signal(exc)
            export_returncode = sig_rc
            export_ok = False
            export_metadata_only = False
            export_error = f"`colab log` interrupted by {sig_name}"
            return
        export_returncode = log_res.returncode
        if log_res.stderr.strip():
            export_stderr_tail = log_res.stderr.strip()[-1000:]
        if log_res.returncode == 0 and not nb_path.is_symlink() and nb_path.is_file():
            nb_endpoint, has_code_cell = inspect_exported_notebook(nb_path)
            if nb_endpoint:
                created_endpoint = nb_endpoint
            if has_code_cell:
                proof_notebook = nb_path
                metadata_notebook = None
                export_ok = True
                export_metadata_only = False
            else:
                proof_notebook = None
                metadata_notebook = nb_path
                export_ok = False
                export_metadata_only = True
                export_error = (
                    f"Exported notebook {nb_path} contains only session metadata and no "
                    "workload code cell (incomplete proof)."
                )
        else:
            export_ok = False
            export_metadata_only = False
            if log_res.returncode != 0:
                diag = log_res.stderr.strip() or log_res.stdout.strip() or "no diagnostic output"
                export_error = f"`colab log` failed (rc={log_res.returncode}): {diag}"
            else:
                export_error = f"`colab log` exited 0 without creating notebook at {nb_path}"

    def _prepare_recovery_store(
        rec_session: str, endpoint: str
    ) -> Tuple[Path, List[str], Optional[str]]:
        nonlocal recovery_config_path
        rec_path = out_dir / f"colab-recovery.{rec_session}.json"
        rec_cmd = build_base_cmd(cfg, caps, rec_path)
        try:
            _write_recovery_session_store(rec_path, rec_session, endpoint)
            recovery_config_path = rec_path
            return rec_path, rec_cmd, None
        except OSError as exc:
            # If the artifact volume is full or read-only, fall back to a private tempdir
            # recovery store so `colab stop` can still release a billable TPU runtime.
            try:
                env_tmp = os.environ.get("TMPDIR")
                tmp_root = Path(env_tmp if env_tmp else tempfile.gettempdir())
                canonical_tmp = _resolve_and_validate_existing_dir(tmp_root)
                fb_dir = Path(
                    tempfile.mkdtemp(
                        prefix="freetpu-colab-recovery-", dir=str(canonical_tmp)
                    )
                )
                os.chmod(fb_dir, 0o700)
                fb_path = fb_dir / f"colab-recovery.{rec_session}.json"
                _write_recovery_session_store(fb_path, rec_session, endpoint)
                recovery_config_path = fb_path
                return fb_path, build_base_cmd(cfg, caps, fb_path), None
            except (UntrustedPathError, OSError):
                return rec_path, rec_cmd, str(exc)
        except UntrustedPathError as exc:
            return rec_path, rec_cmd, str(exc)

    def _run_forced_stop(sess_name: str) -> Tuple[CommandResult, Optional[str]]:
        # Avoid invoking `colab stop` directly against a pre-populated user `--config`
        # because Colab CLI's `stop` runs `sync_sessions()` and prunes other accounts' entries.
        if has_existing_store_entries and created_endpoint is not None:
            _, rec_base, seed_err = _prepare_recovery_store(sess_name, created_endpoint)
            if seed_err is None:
                return _run_stop_cmd(sess_name, cmd_prefix=rec_base)
        return _run_stop_cmd(sess_name)

    prev_sigterm_handler = None
    sigterm_installed = False
    if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
        def _handle_sigterm(signum: int, frame: Any) -> None:
            nonlocal terminated
            del signum, frame
            terminated = True
            cleanup_wake_event.set()
            if not in_cleanup:
                raise TerminationSignal("Received SIGTERM")

        prev_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)
        sigterm_installed = True

    try:
        for idx, tpu in enumerate(cfg.tpus):
            if terminated:
                final_rc = 143
                break
            sess = f"{session_base}-{tpu}" if idx > 0 else session_base
            attempted_sessions.append(sess)
            run_cmd = [
                *run_base_cmd,
                "run",
                "-s",
                sess,
                "--tpu",
                tpu.upper(),
                "--timeout",
                str(cfg.timeout_s),
                "--",
                str(script_file),
                *cfg.script_args,
            ]
            in_cleanup = False
            try:
                res = runner(run_cmd)
            except (KeyboardInterrupt, TerminationSignal, CommandCancelledError) as cancel_exc:
                in_cleanup = True
                if isinstance(cancel_exc, CommandCancelledError):
                    is_sigterm = cancel_exc.signal_name == "SIGTERM"
                    cancel_rc = cancel_exc.returncode
                    partial_stdout = cancel_exc.stdout
                    partial_stderr = cancel_exc.stderr
                    cancel_probe_text = cancel_exc.captured_probe_text
                    cancel_stdout_written = cancel_exc.stdout_log_written
                    cancel_stderr_written = cancel_exc.stderr_log_written
                    if cancel_exc.untrusted_log_error:
                        untrusted_target_error = cancel_exc.untrusted_log_error
                    if cancel_exc.log_write_error and artifact_write_error is None:
                        artifact_write_error = cancel_exc.log_write_error
                else:
                    is_sigterm = isinstance(cancel_exc, TerminationSignal)
                    cancel_rc = 143 if is_sigterm else 130
                    partial_stdout = ""
                    partial_stderr = ""
                    cancel_probe_text = None
                    cancel_stdout_written = False
                    cancel_stderr_written = False

                if is_sigterm:
                    terminated = True
                else:
                    interrupted = True
                maybe_allocated = True
                final_rc = cancel_rc
                if chosen_tpu is None:
                    chosen_tpu = tpu
                    chosen_session = sess

                # Recover the allocated endpoint independently of notebook export before stop/reconcile.
                _recover_session_endpoint(sess)

                cancel_note = (
                    "[colab_tpu_bench] Terminated by supervisor (SIGTERM)."
                    if is_sigterm
                    else "[colab_tpu_bench] Interrupted by operator (KeyboardInterrupt)."
                )
                combined_cancel_stderr = (
                    f"{partial_stderr.rstrip()}\n{cancel_note}\n"
                    if partial_stderr.strip()
                    else f"{cancel_note}\n"
                )
                if "[colab] session ready" in partial_stderr.lower():
                    ready_seen_any = True
                parsed_partial = (
                    extract_probe_json(cancel_probe_text)
                    if cancel_probe_text is not None
                    else None
                )
                if parsed_partial is None:
                    parsed_partial = extract_probe_json(partial_stdout)
                if parsed_partial is not None:
                    probe_payload = parsed_partial

                attempts.append(
                    {
                        "tpu": tpu,
                        "session": sess,
                        "returncode": cancel_rc,
                        "stderr_tail": combined_cancel_stderr.strip()[-1000:],
                    }
                )
                # Reconcile the active TPU session first using the attempted session name and
                # isolated config, then preserve any already-emitted workload output and log proof.
                forced_stop["attempted"] = True
                forced_stop["session"] = sess
                stop_res, stop_err = _run_forced_stop(sess)
                forced_stop["returncode"] = stop_res.returncode
                if stop_res.stderr.strip():
                    forced_stop["stderr_tail"] = stop_res.stderr.strip()[-1000:]
                forced_stop["error"] = stop_err

                if untrusted_target_error is None:
                    try:
                        if partial_stdout and not cancel_stdout_written:
                            _write_text_nofollow(out_dir / f"{sess}.stdout.txt", partial_stdout)
                        if combined_cancel_stderr and not cancel_stderr_written:
                            _write_text_nofollow(
                                out_dir / f"{sess}.stderr.txt", combined_cancel_stderr
                            )
                    except UntrustedPathError as exc:
                        untrusted_target_error = str(exc)
                    except OSError as exc:
                        if artifact_write_error is None:
                            artifact_write_error = f"Failed writing local run logs: {exc}"

                if untrusted_target_error is None:
                    _export_session_log(sess)
                break
            finally:
                in_cleanup = True

            if res.untrusted_log_error:
                untrusted_target_error = res.untrusted_log_error
            if res.log_write_error and artifact_write_error is None:
                artifact_write_error = res.log_write_error

            # Recover the allocated endpoint from Colab CLI's session_created history event
            # independently of notebook export so an export or disk-full failure cannot hide a TPU.
            _recover_session_endpoint(sess)

            attempt_rec = {
                "tpu": tpu,
                "session": sess,
                "returncode": res.returncode,
                "stderr_tail": res.stderr.strip()[-1000:],
            }
            attempts.append(attempt_rec)

            parsed_probe = (
                extract_probe_json(res.captured_probe_text)
                if res.captured_probe_text is not None
                else None
            )
            if parsed_probe is None:
                parsed_probe = extract_probe_json(res.stdout)
            if parsed_probe is not None:
                probe_payload = parsed_probe

            ready_seen = "[colab] session ready" in res.stderr.lower()
            if ready_seen:
                ready_seen_any = True
            wall_timeout_seen = is_wall_timeout_error(res)
            timeout_seen = (
                is_allocation_timeout_error(res.returncode, res.stderr, timed_out=res.timed_out)
                or (res.returncode != 0 and not ready_seen and bool(res.pre_ready_uncertain))
            )
            if (
                not timeout_seen
                and res.returncode != 0
                and not ready_seen
                and res.elapsed_s is not None
                and res.elapsed_s >= effective_request_timeout
            ):
                # A pre-READY failure whose duration reached the HTTP request timeout may have
                # suffered a timed-out POST followed by an internal CLI retry returning 412;
                # treat the outcome as uncertain and block fallback.
                timeout_seen = True
            if (
                not timeout_seen
                and not retry_signal_observable
                and res.returncode != 0
                and not ready_seen
                and is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)
            ):
                # Without `--logtostderr` while internal allocation retries are enabled, a final
                # 412 refusal cannot rule out an earlier retried 502/503/504/ReadTimeout.
                timeout_seen = True
            confirmed_pre_alloc = (
                res.returncode != 0
                and not ready_seen
                and not wall_timeout_seen
                and not timeout_seen
                and (
                    is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)
                    or _contains_oauth_prompt_failure(res)
                )
            )
            if (
                res.returncode == 0
                or ready_seen
                or wall_timeout_seen
                or timeout_seen
                or not confirmed_pre_alloc
            ):
                maybe_allocated = True

            # If `subprocess` killed `colab run` on wall-clock timeout, its own
            # `finally: _teardown(...)` handler never ran; explicitly stop the session
            # using the exact same isolated `--config` before exporting logs or checking sessions.
            if wall_timeout_seen:
                wall_timed_out = True
                forced_stop["attempted"] = True
                forced_stop["session"] = sess
                stop_res, stop_err = _run_forced_stop(sess)
                forced_stop["returncode"] = stop_res.returncode
                if stop_res.stderr.strip():
                    forced_stop["stderr_tail"] = stop_res.stderr.strip()[-1000:]
                forced_stop["error"] = stop_err

            if untrusted_target_error is not None:
                break

            try:
                if res.stdout and not res.stdout_log_written:
                    _write_text_nofollow(out_dir / f"{sess}.stdout.txt", res.stdout)
                if res.returncode != 0 and res.stderr and not res.stderr_log_written:
                    _write_text_nofollow(out_dir / f"{sess}.stderr.txt", res.stderr)
            except UntrustedPathError as exc:
                untrusted_target_error = str(exc)
                break
            except OSError as exc:
                if artifact_write_error is None:
                    artifact_write_error = f"Failed writing local run logs: {exc}"

            if res.returncode == 0:
                chosen_tpu = tpu
                chosen_session = sess
                if generated_script:
                    probe_error = validate_builtin_probe(probe_payload)
                    final_rc = 1 if probe_error is not None else 0
                else:
                    final_rc = 0
            else:
                final_rc = res.returncode

            if res.returncode == 0 or ready_seen or wall_timeout_seen:
                if chosen_tpu is None:
                    chosen_tpu = tpu
                    chosen_session = sess
                _export_session_log(sess)
                if untrusted_target_error is not None:
                    break

            if res.returncode == 0:
                break

            if timeout_seen:
                allocation_timed_out = True
                break

            if wall_timeout_seen:
                break

            if terminated:
                final_rc = 143
                break

            has_more = idx + 1 < len(cfg.tpus)
            if not (has_more and is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)):
                break

            # Before falling back to a secondary accelerator, verify no assignment from the
            # failed attempt materialized on the server during an internal CLI retry.
            pre_fb_res = runner([*list_base_cmd, "sessions"])
            if pre_fb_res.returncode != 0 or _has_colab_auth_failure(pre_fb_res):
                maybe_allocated = True
                allocation_timed_out = True
                break
            _, pre_fb_endpoints = parse_sessions_output(pre_fb_res.stdout)
            if pre_fb_endpoints - endpoints_before:
                maybe_allocated = True
                allocation_timed_out = True
                break

        # 3. Verify server-side TPU assignment release whenever an allocation may have occurred,
        #    applying bounded retries to both active-assignment leaks and transient listing failures
        #    using a dedicated per-run recovery store (never mutating another worker's `--config`).
        in_cleanup = True
        teardown_verified = not maybe_allocated
        leaked_endpoints: List[str] = []
        leaked_sessions: List[str] = []
        ambiguous_endpoints: List[str] = []
        other_new_endpoints: List[str] = []
        teardown_error: Optional[str] = None

        def _inspect_active_sessions() -> Tuple[
            Optional[str], bool, List[str], List[str], List[str], List[str]
        ]:
            try:
                sess_res = runner([*list_base_cmd, "sessions"])
            except (TerminationSignal, CommandCancelledError) as exc:
                sig_rc, sig_name = _record_cleanup_signal(exc)
                return (
                    f"Post-run `colab sessions` interrupted by {sig_name} (rc={sig_rc})",
                    False,
                    [],
                    [],
                    [],
                    [],
                )
            is_auth_err = _has_colab_auth_failure(sess_res)
            if sess_res.returncode != 0 or is_auth_err:
                diag = sess_res.stderr.strip() or sess_res.stdout.strip()
                if sess_res.returncode != 0:
                    return (
                        f"Post-run `colab sessions` failed (rc={sess_res.returncode}): {diag}",
                        is_auth_err,
                        [],
                        [],
                        [],
                        [],
                    )
                return (
                    f"Post-run `colab sessions` exited 0 with authentication error: {diag}",
                    True,
                    [],
                    [],
                    [],
                    [],
                )
            names_after, endpoints_after = parse_sessions_output(sess_res.stdout)
            new_endpoints = sorted(endpoints_after - endpoints_before)
            cur_leaked_eps: List[str] = []
            cur_other_eps: List[str] = []
            cur_ambig_eps: List[str] = []
            if created_endpoint is not None:
                if created_endpoint in endpoints_after:
                    cur_leaked_eps.append(created_endpoint)
                cur_other_eps = [ep for ep in new_endpoints if ep != created_endpoint]
            else:
                cur_ambig_eps = new_endpoints
            cur_leaked_sess = [
                sname
                for sname in attempted_sessions
                if sname in names_after and sname not in names_before
            ]
            return None, False, cur_leaked_eps, cur_leaked_sess, cur_ambig_eps, cur_other_eps

        def _find_original_store_session(endpoint: str) -> Optional[str]:
            """Return a session name still mapped to `endpoint` in `isolated_config`, if any."""
            candidates = list(
                dict.fromkeys([*leaked_sessions, recovery_session, *attempted_sessions])
            )
            for cand in candidates:
                if read_endpoint_from_session_store(isolated_config, cand) == endpoint:
                    return cand
            if leaked_sessions:
                return leaked_sessions[0]
            return None

        if maybe_allocated:
            if created_endpoint is None:
                for sname in reversed(attempted_sessions):
                    if _recover_session_endpoint(sname):
                        break
            (
                query_err,
                query_is_auth_err,
                leaked_endpoints,
                leaked_sessions,
                ambiguous_endpoints,
                other_new_endpoints,
            ) = _inspect_active_sessions()
            recovery_session = (
                chosen_session
                or (leaked_sessions[0] if leaked_sessions else None)
                or (attempted_sessions[-1] if attempted_sessions else session_base)
            )
            max_retries = max(int(cfg.teardown_retries), 0)
            retry_idx = 0

            while True:
                if query_err is None and not leaked_endpoints and not leaked_sessions:
                    if ambiguous_endpoints:
                        teardown_verified = False
                        teardown_error = (
                            "Run endpoint is unknown and new Colab assignment(s) appeared after run: "
                            f"ambiguous_endpoints={ambiguous_endpoints}"
                        )
                    elif created_endpoint is None and (
                        allocation_timed_out or ((interrupted or terminated) and not ready_seen_any)
                    ):
                        recheck_cmd = shlex.join([*list_base_cmd, "sessions"])
                        teardown_verified = False
                        teardown_error = (
                            "Run ended before an allocation endpoint was confirmed; an in-flight server "
                            "allocation may still materialize after this snapshot. Re-check active runtimes "
                            f"(run: `{recheck_cmd}`) and stop any newly assigned TPU; isolated config "
                            f"preserved at {isolated_config}."
                        )
                    else:
                        teardown_verified = True
                        teardown_error = None
                    break

                if query_is_auth_err or retry_idx >= max_retries:
                    teardown_verified = False
                    seed_failure_detail: Optional[str] = None
                    preserved_path: Optional[Path] = None
                    if created_endpoint is not None and (query_err is not None or leaked_endpoints):
                        rec_path, rec_base, final_seed_err = _prepare_recovery_store(
                            recovery_session, created_endpoint
                        )
                        if final_seed_err is None:
                            recovery_command = shlex.join(
                                [*rec_base, "stop", "-s", recovery_session]
                            )
                            preserved_path = rec_path
                        else:
                            orig_sess = _find_original_store_session(created_endpoint)
                            if orig_sess is not None:
                                recovery_command = shlex.join(
                                    [*base_cmd, "stop", "-s", orig_sess]
                                )
                                preserved_path = isolated_config
                            else:
                                recovery_command = None
                                seed_failure_detail = (
                                    f"failed to seed recovery store ({final_seed_err}) and "
                                    f"original session store {isolated_config} has no mapping for "
                                    f"endpoint `{created_endpoint}`; stop it manually in Colab "
                                    "Manage Sessions (https://colab.research.google.com)"
                                )
                    elif leaked_sessions:
                        recovery_command = shlex.join([*base_cmd, "stop", "-s", recovery_session])
                        preserved_path = isolated_config
                    else:
                        preserved_path = isolated_config

                    if query_err is not None:
                        if recovery_command and preserved_path is not None:
                            teardown_error = (
                                f"{query_err}; per-run recovery config preserved at {preserved_path} "
                                f"(run: `{recovery_command}`)"
                            )
                        elif seed_failure_detail is not None:
                            teardown_error = f"{query_err}; {seed_failure_detail}"
                        else:
                            teardown_error = query_err
                    else:
                        if recovery_command and preserved_path is not None:
                            teardown_error = (
                                "Active Colab assignment/session remained after release retries: "
                                f"endpoints={leaked_endpoints}, sessions={leaked_sessions}; "
                                f"per-run recovery config preserved at {preserved_path} "
                                f"(run: `{recovery_command}`)"
                            )
                        elif seed_failure_detail is not None:
                            teardown_error = (
                                "Active Colab assignment/session remained after release retries: "
                                f"endpoints={leaked_endpoints}, sessions={leaked_sessions}; "
                                f"{seed_failure_detail}"
                            )
                        else:
                            teardown_error = (
                                "Active Colab assignment/session remained after release retries: "
                                f"endpoints={leaked_endpoints}, sessions={leaked_sessions}"
                            )
                    break

                retry_idx += 1
                if (
                    (retry_idx > 1 or query_err is not None)
                    and cfg.teardown_retry_delay_s > 0
                    and not terminated
                ):
                    try:
                        cleanup_wake_event.wait(timeout=cfg.teardown_retry_delay_s)
                    except TerminationSignal:
                        terminated = True
                        final_rc = 143
                    except KeyboardInterrupt:
                        interrupted = True
                        final_rc = 130
                        retry_idx = max_retries
                        continue

                if created_endpoint is not None and (leaked_endpoints or query_err is not None):
                    rec_path, rec_base, seed_err = _prepare_recovery_store(
                        recovery_session, created_endpoint
                    )
                    stopped_via_orig: Optional[str] = None
                    if seed_err is None:
                        retry_res, retry_err = _run_stop_cmd(recovery_session, cmd_prefix=rec_base)
                        release_retries.append(
                            {
                                "attempt": retry_idx,
                                "session": recovery_session,
                                "endpoint": created_endpoint,
                                "recovery_config": str(rec_path),
                                "seed_error": None,
                                "returncode": retry_res.returncode,
                                "error": retry_err,
                                "stderr_tail": retry_res.stderr.strip()[-1000:]
                                if retry_res.stderr.strip()
                                else None,
                            }
                        )
                    else:
                        orig_sess = _find_original_store_session(created_endpoint)
                        if orig_sess is not None:
                            stopped_via_orig = orig_sess
                            retry_res, retry_err = _run_stop_cmd(orig_sess, cmd_prefix=base_cmd)
                            release_retries.append(
                                {
                                    "attempt": retry_idx,
                                    "session": orig_sess,
                                    "endpoint": created_endpoint,
                                    "recovery_config": str(isolated_config),
                                    "seed_error": seed_err,
                                    "returncode": retry_res.returncode,
                                    "error": retry_err,
                                    "stderr_tail": retry_res.stderr.strip()[-1000:]
                                    if retry_res.stderr.strip()
                                    else None,
                                }
                            )
                        else:
                            release_retries.append(
                                {
                                    "attempt": retry_idx,
                                    "session": recovery_session,
                                    "endpoint": created_endpoint,
                                    "recovery_config": None,
                                    "seed_error": seed_err,
                                    "returncode": None,
                                    "error": (
                                        f"Skipped `colab stop` with unseeded config: {seed_err}; "
                                        f"original store {isolated_config} has no mapping for {created_endpoint}"
                                    ),
                                    "stderr_tail": None,
                                }
                            )
                    for extra_sess in list(dict.fromkeys(leaked_sessions)):
                        if extra_sess == recovery_session and seed_err is None:
                            continue
                        if extra_sess == stopped_via_orig:
                            continue
                        extra_res, extra_err = _run_stop_cmd(extra_sess)
                        release_retries.append(
                            {
                                "attempt": retry_idx,
                                "session": extra_sess,
                                "endpoint": created_endpoint,
                                "recovery_config": str(isolated_config),
                                "seed_error": None,
                                "returncode": extra_res.returncode,
                                "error": extra_err,
                                "stderr_tail": extra_res.stderr.strip()[-1000:]
                                if extra_res.stderr.strip()
                                else None,
                            }
                        )
                elif leaked_sessions:
                    for target_sess in list(dict.fromkeys(leaked_sessions)):
                        retry_res, retry_err = _run_stop_cmd(target_sess)
                        release_retries.append(
                            {
                                "attempt": retry_idx,
                                "session": target_sess,
                                "endpoint": created_endpoint,
                                "recovery_config": str(isolated_config),
                                "seed_error": None,
                                "returncode": retry_res.returncode,
                                "error": retry_err,
                                "stderr_tail": retry_res.stderr.strip()[-1000:]
                                if retry_res.stderr.strip()
                                else None,
                            }
                        )
                else:
                    release_retries.append(
                        {
                            "attempt": retry_idx,
                            "session": recovery_session,
                            "endpoint": None,
                            "recovery_config": None,
                            "seed_error": None,
                            "returncode": None,
                            "error": query_err,
                            "stderr_tail": None,
                        }
                    )

                (
                    query_err,
                    query_is_auth_err,
                    leaked_endpoints,
                    leaked_sessions,
                    ambiguous_endpoints,
                    other_new_endpoints,
                ) = _inspect_active_sessions()

        if untrusted_target_error is not None:
            return 2, {
                "status": "untrusted_out_dir",
                "error": untrusted_target_error,
                "forced_stop": forced_stop,
                "release_retries": release_retries,
                "recovery_config": str(recovery_config_path) if recovery_config_path else None,
                "recovery_command": recovery_command,
                "teardown_verified": teardown_verified,
                "leaked_endpoints": leaked_endpoints,
                "leaked_sessions": leaked_sessions,
                "ambiguous_endpoints": ambiguous_endpoints,
                "teardown_error": teardown_error,
            }

        if terminated:
            status = "terminated" if teardown_verified else "terminated_unreconciled"
            final_rc = 143
        elif interrupted:
            status = "interrupted" if teardown_verified else "interrupted_unreconciled"
            final_rc = 130
        elif allocation_timed_out and not teardown_verified:
            status = "allocation_timeout_unreconciled"
            if final_rc == 0:
                final_rc = 1
        elif allocation_timed_out:
            status = "allocation_timeout"
            if final_rc == 0:
                final_rc = 1
        elif not teardown_verified:
            if ambiguous_endpoints and not leaked_endpoints and not leaked_sessions:
                status = "teardown_ambiguous"
            else:
                status = "teardown_verification_failed"
            if final_rc == 0:
                final_rc = 1
        elif wall_timed_out:
            status = "wall_timeout"
            if final_rc == 0:
                final_rc = 1
        elif probe_error is not None:
            status = "probe_validation_failed"
            if final_rc == 0:
                final_rc = 1
        elif final_rc == 0 and export_attempted and not export_ok:
            status = "proof_export_failed"
            final_rc = 1
        elif artifact_write_error is not None:
            status = "artifact_write_failed"
            if final_rc == 0:
                final_rc = 1
        elif final_rc == 0:
            status = "ok"
        else:
            status = "failed"

        run_summary_path = out_dir / f"{chosen_session or session_base}.summary.json"
        summary = {
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "colab_cli_version": cli_version,
            "effective_wall_timeout_s": effective_wall_timeout,
            "effective_request_timeout_s": effective_request_timeout,
            "isolated_config": str(isolated_config),
            "recovery_config": str(recovery_config_path) if recovery_config_path else None,
            "run_summary": str(run_summary_path),
            "usage_before": asdict(usage),
            "chosen_tpu": chosen_tpu,
            "chosen_session": chosen_session,
            "created_endpoint": created_endpoint,
            "proof_notebook": str(proof_notebook) if proof_notebook else None,
            "metadata_notebook": str(metadata_notebook) if metadata_notebook else None,
            "proof_export": {
                "attempted": export_attempted,
                "ok": export_ok,
                "metadata_only": export_metadata_only,
                "returncode": export_returncode,
                "error": export_error,
                "stderr_tail": export_stderr_tail,
            },
            "forced_stop": forced_stop,
            "release_retries": release_retries,
            "recovery_command": recovery_command,
            "teardown_verified": teardown_verified,
            "leaked_endpoints": leaked_endpoints,
            "leaked_sessions": leaked_sessions,
            "ambiguous_endpoints": ambiguous_endpoints,
            "other_new_endpoints": other_new_endpoints,
            "teardown_error": teardown_error,
            "probe_error": probe_error,
            "artifact_write_error": artifact_write_error,
            "attempts": attempts,
            "probe": probe_payload,
        }
        summary_json_text = json.dumps(summary, indent=2) + "\n"
        try:
            _write_text_nofollow(run_summary_path, summary_json_text)
            _write_text_nofollow(out_dir / "summary.json", summary_json_text)
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        except OSError as exc:
            summary["artifact_write_error"] = summary["artifact_write_error"] or str(exc)
            if summary["status"] == "ok":
                summary["status"] = "artifact_write_failed"
                final_rc = 1
        return final_rc, summary
    finally:
        for _tmp_list_file in (listing_config, Path(f"{listing_config}.lock")):
            try:
                _tmp_list_file.unlink(missing_ok=True)
            except OSError:
                pass
        if sigterm_installed:
            signal.signal(signal.SIGTERM, prev_sigterm_handler)


def run_self_test() -> int:
    """Offline self-test verifying safety, CU guards, fallback, teardown, and edge cases."""
    with tempfile.TemporaryDirectory(prefix="freetpu-colab-selftest-") as tmp:
        tmp_path = Path(tmp)
        os.chmod(tmp_path, 0o700)

        def base_help_usage_sessions(
            cmd_list: List[str],
            *,
            balance: str = "500.00",
            active: str = "0",
            sessions_out: str = "[colab] No active sessions found on server.\n",
        ) -> Optional[CommandResult]:
            if "--help" in cmd_list:
                return CommandResult(
                    0, "Usage: colab --auth --config --logtostderr --no-public-api\n", ""
                )
            if "version" in cmd_list:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd_list:
                return CommandResult(
                    0,
                    f"Current balance: {balance} compute units\nUsage rate: 0.00/hr\nActive assignments: {active}\n",
                    "",
                )
            if "sessions" in cmd_list:
                return CommandResult(0, sessions_out, "")
            return None

        # 1. Signed usage parser & sessions parser checks.
        parsed = parse_usage_output(
            "Current balance: 1995.50 compute units\nUsage rate: 2.31/hr\nActive assignments: 1\n"
        )
        assert parsed.balance_cu == 1995.50
        assert parsed.usage_rate_per_hour == 2.31
        assert parsed.active_assignments == 1

        parsed_neg = parse_usage_output(
            "Current balance: -2.50 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n"
        )
        assert parsed_neg.balance_cu == -2.50

        s_names, s_eps = parse_sessions_output(
            "[s1] m-s-abc123 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n"
            "[?] m-s-orphan | Hardware: CPU | Shape: STANDARD | Variant: DEFAULT\n"
        )
        assert s_names == {"s1"}
        assert s_eps == {"m-s-abc123", "m-s-orphan"}

        # 2. Signed/unparseable CU balance guard aborts before `colab run`.
        seen_cmds: List[List[str]] = []

        def low_or_bad_cu_runner(balance_line: str) -> CommandRunner:
            def _r(cmd: Sequence[str]) -> CommandResult:
                seen_cmds.append(list(cmd))
                if "--help" in cmd:
                    return CommandResult(0, "Usage: [OPTIONS] --config --auth\n", "")
                if "version" in cmd:
                    return CommandResult(0, "Version: 0.8.0\n", "")
                if "usage" in cmd:
                    return CommandResult(0, balance_line, "")
                raise AssertionError(f"Unexpected command after preflight abort: {cmd}")

            return _r

        out_low = tmp_path / "low-cu"
        rc_low, sum_low = run_benchmark(
            BenchConfig(min_cu_balance=5.0, out_dir=out_low),
            runner=low_or_bad_cu_runner(
                "Current balance: -1.50 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n"
            ),
        )
        assert rc_low == 1
        assert sum_low["status"] == "cu_budget_insufficient"
        assert sum_low["usage"]["balance_cu"] == -1.50

        rc_unp, sum_unp = run_benchmark(
            BenchConfig(min_cu_balance=5.0, out_dir=tmp_path / "unparseable-cu"),
            runner=low_or_bad_cu_runner("Current balance: unavailable\n"),
        )
        assert rc_unp == 1
        assert sum_unp["status"] == "usage_parse_failed"
        assert not any("run" in c for c in seen_cmds)

        # 3. v6e1 412 fallback -> v5e1 success + log export + `--` separator + unique session + clean sessions check.
        fallback_cmds: List[List[str]] = []
        iso_cfg = Path("bare-sessions.json")
        orig_cwd = os.getcwd()
        try:
            cwd_box = tmp_path / "cwd-box"
            cwd_box.mkdir(mode=0o700)
            os.chdir(cwd_box)

            def fallback_runner(cmd: Sequence[str]) -> CommandResult:
                cmd_list = list(cmd)
                fallback_cmds.append(cmd_list)
                base_res = base_help_usage_sessions(cmd_list)
                if base_res is not None:
                    return base_res
                if "run" in cmd_list:
                    # Verify `--` separates script path/args from `colab run` options
                    run_idx = cmd_list.index("run")
                    sep_idx = cmd_list.index("--")
                    assert sep_idx > run_idx
                    assert cmd_list[sep_idx + 1].endswith("colab_tpu_probe_payload.py")
                    if "V6E1" in cmd_list:
                        return CommandResult(
                            1,
                            "",
                            "[colab] Allocation refused (precondition failed): REASON_NO_BACKENDS (412)",
                        )
                    if "V5E1" in cmd_list:
                        payload = json.dumps(
                            {
                                "schema": "freetpu.colab_tpu_probe.v1",
                                "jax": {
                                    "default_backend": "tpu",
                                    "device_count": 1,
                                    "devices": [{"platform": "tpu", "device_kind": "TPU v5 lite"}],
                                },
                                "benchmark": {
                                    "hardware": "colab-tpu",
                                    "silicon": "TPU v5 lite",
                                    "hw_us": 123.4,
                                    "gflops": 270.0,
                                },
                            }
                        )
                        return CommandResult(
                            0,
                            f"{PROBE_JSON_BEGIN}\n{payload}\n{PROBE_JSON_END}\n",
                            "[colab] Session READY.\n[colab] Session stopped.\n",
                        )
                if "log" in cmd_list:
                    out_idx = cmd_list.index("-o") + 1
                    nb = {
                        "nbformat": 4,
                        "cells": [
                            {
                                "cell_type": "markdown",
                                "source": "**Session Created**: now\n- Endpoint: `tpu-v5e1-ep1`\n",
                            },
                            {
                                "cell_type": "code",
                                "source": "print('probe')\n",
                            },
                        ],
                    }
                    Path(cmd_list[out_idx]).write_text(json.dumps(nb), encoding="utf-8")
                    return CommandResult(0, "Exported notebook.\n", "")
                raise AssertionError(f"Unexpected command: {cmd_list}")

            rc_fb, sum_fb = run_benchmark(
                BenchConfig(
                    config_path=iso_cfg,
                    out_dir=tmp_path / "fallback",
                    session_name="tpu-check",
                    script_args=("--keep", "--timeout", "0"),
                ),
                runner=fallback_runner,
            )
            assert rc_fb == 0
            assert sum_fb["status"] == "ok"
            assert sum_fb["chosen_tpu"] == "v5e1"
            assert sum_fb["teardown_verified"] is True
            assert sum_fb["created_endpoint"] == "tpu-v5e1-ep1"
            assert sum_fb["chosen_session"].startswith("tpu-check-")
            assert sum_fb["chosen_session"] != "tpu-check"
            assert os.path.isabs(sum_fb["isolated_config"])
            assert sum_fb["isolated_config"].endswith("bare-sessions.json")
            assert len(sum_fb["attempts"]) == 2
            assert sum_fb["probe"]["benchmark"]["silicon"] == "TPU v5 lite"
            assert sum_fb["proof_notebook"] is not None
            assert Path(sum_fb["proof_notebook"]).is_file()
            for c in fallback_cmds:
                if "--help" not in c and "version" not in c:
                    assert "--config" in c
                    if "sessions" in c:
                        cfg_val = c[c.index("--config") + 1]
                        assert Path(cfg_val).name.startswith("colab-sessions-list.")
                    else:
                        assert sum_fb["isolated_config"] in c
                    assert "--no-public-api" in c
                    if "run" in c:
                        assert "--logtostderr" in c

            # Repeat call with identical --session must get a distinct unique session name
            rc_fb2, sum_fb2 = run_benchmark(
                BenchConfig(
                    config_path=iso_cfg,
                    out_dir=tmp_path / "fallback2",
                    session_name="tpu-check",
                ),
                runner=fallback_runner,
            )
            assert rc_fb2 == 0
            assert sum_fb2["chosen_session"].split("-v5e1")[0] != sum_fb["chosen_session"].split("-v5e1")[0]
        finally:
            os.chdir(orig_cwd)

        # 4. Leaked TPU endpoint that survives release retries fails teardown verification
        #    and preserves the recovery mapping in `isolated_config` (never returns ok).
        sess_calls = {"n": 0}
        leak_stop_calls = {"n": 0}

        def leaked_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                sess_calls["n"] += 1
                if sess_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[?] tpu-v5e1-ep1 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            if "stop" in cmd_list:
                leak_stop_calls["n"] += 1
                return CommandResult(1, "", "unassign failed: 503 Service Unavailable")
            return fallback_runner(cmd_list)

        rc_leak, sum_leak = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                out_dir=tmp_path / "leaked",
                teardown_retries=2,
                teardown_retry_delay_s=0.0,
            ),
            runner=leaked_runner,
        )
        assert rc_leak == 1
        assert sum_leak["status"] == "teardown_verification_failed"
        assert sum_leak["teardown_verified"] is False
        assert "tpu-v5e1-ep1" in sum_leak["leaked_endpoints"]
        assert leak_stop_calls["n"] == 2
        assert len(sum_leak["release_retries"]) == 2
        assert sum_leak["recovery_command"] is not None
        assert sum_leak["recovery_config"] is not None
        preserved_cfg = json.loads(Path(sum_leak["recovery_config"]).read_text(encoding="utf-8"))
        assert preserved_cfg[sum_leak["chosen_session"]]["endpoint"] == "tpu-v5e1-ep1"

        # 5. Built-in CPU-fallback probe payload (default_backend="cpu") fails validation.
        def cpu_probe_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                payload = json.dumps(
                    {
                        "schema": "freetpu.colab_tpu_probe.v1",
                        "jax": {
                            "default_backend": "cpu",
                            "device_count": 1,
                            "devices": [{"platform": "cpu"}],
                        },
                        "benchmark": {"hardware": "colab-tpu", "hw_us": 50.0, "gflops": 100.0},
                    }
                )
                return CommandResult(0, f"{PROBE_JSON_BEGIN}\n{payload}\n{PROBE_JSON_END}\n", "")
            if "log" in cmd_list:
                return CommandResult(1, "", "no log")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_cpu, sum_cpu = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "cpu-fallback"),
            runner=cpu_probe_runner,
        )
        assert rc_cpu == 1
        assert sum_cpu["status"] == "probe_validation_failed"

        # 6. Allocation-stage ReadTimeout does NOT fall back to v5e1, preventing double allocation.
        timeout_cmds: List[List[str]] = []

        def timeout_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            timeout_cmds.append(cmd_list)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(
                    1,
                    "",
                    "[colab] Creating session 'x'...\n"
                    "ReadTimeout: HTTPSConnectionPool(host='colab.research.google.com', port=443): "
                    "Read timed out. (read timeout=45.0)",
                )
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_to, sum_to = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "timeout"),
            runner=timeout_runner,
        )
        assert rc_to == 1
        assert len(sum_to["attempts"]) == 1
        assert sum_to["attempts"][0]["tpu"] == "v6e1"
        assert sum_to["status"] == "allocation_timeout_unreconciled"
        assert sum_to["teardown_verified"] is False
        assert "re-check" in (sum_to["teardown_error"] or "").lower()

        assert not is_capacity_or_quota_error(
            1,
            "HTTP 412 / REASON_NO_BACKENDS",
            "[colab] Session READY\nReadTimeout",
        )
        assert not is_capacity_or_quota_error(1, "", "ReadTimeout: Read timed out.")

        # 7. --public-api is explicitly rejected because `colab run` still uses `/tun/m/*`.
        rc_pub, sum_pub = run_benchmark(
            BenchConfig(use_no_public_api=False, dry_run=True, out_dir=tmp_path / "pub-bad"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_pub == 2
        assert sum_pub["status"] == "unsupported_public_api_mode"

        # 8. `--env` inside script_args is rejected before building commands.
        rc_env, sum_env = run_benchmark(
            BenchConfig(dry_run=True, out_dir=tmp_path / "bad-env", script_args=("--env=FOO=1",)),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_env == 2
        assert sum_env["status"] == "unsupported_script_arg"

        # 9. Untrusted world-readable dir and symlinked out_dir / config are rejected.
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir(mode=0o755)
        os.chmod(shared_dir, 0o755)
        rc_sh, sum_sh = run_benchmark(
            BenchConfig(dry_run=True, out_dir=shared_dir),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_sh == 2
        assert sum_sh["status"] == "untrusted_out_dir"

        sym_dir = tmp_path / "sym-out"
        sym_dir.symlink_to(tmp_path)
        rc_sym, sum_sym = run_benchmark(
            BenchConfig(dry_run=True, out_dir=sym_dir),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_sym == 2
        assert sum_sym["status"] == "untrusted_out_dir"

        # 10. OAuth2 non-interactive EOF failure surfaces clear `oauth2_auth_required` error.
        def oauth_eof_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: --auth --config\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            return CommandResult(
                1,
                "",
                "To authorize colab-cli, visit this URL in any browser:\nEOFError: EOF when reading a line",
            )

        rc_oa, sum_oa = run_benchmark(
            BenchConfig(auth="oauth2", out_dir=tmp_path / "oauth-eof"),
            runner=oauth_eof_runner,
        )
        assert rc_oa == 1
        assert sum_oa["status"] == "oauth2_auth_required"

        # 11. Absent `colab` CLI still produces a clean dry-run plan without raising FileNotFoundError.
        def missing_cli_runner(cmd: Sequence[str]) -> CommandResult:
            raise FileNotFoundError(cmd[0])

        rc_dry_missing, sum_dry_missing = run_benchmark(
            BenchConfig(dry_run=True, out_dir=tmp_path / "dry-missing"),
            runner=missing_cli_runner,
        )
        assert rc_dry_missing == 0
        assert sum_dry_missing["status"] == "dry_run"
        assert sum_dry_missing["colab_cli_found"] is False
        assert len(sum_dry_missing["planned_commands"]) == 7

        # 12. Pre-READY non-refusal failure (e.g., local state write OSError) still runs teardown check.
        pre_ready_cmds: List[List[str]] = []

        def pre_ready_fail_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            pre_ready_cmds.append(cmd_list)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(1, "", "PermissionError: cannot save ~/.config/colab-cli/sessions.json")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_pre, sum_pre = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "pre-ready-fail"),
            runner=pre_ready_fail_runner,
        )
        assert rc_pre == 1
        assert sum_pre["status"] == "failed"
        assert sum_pre["teardown_verified"] is True
        assert sum(1 for c in pre_ready_cmds if "sessions" in c) == 2

        # 13. Rich/Typer traceback mentioning `except TooManyAssignmentsError` must NOT trigger v5e1 fallback.
        tb_cmds: List[List[str]] = []

        def rich_403_traceback_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            tb_cmds.append(cmd_list)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(
                    1,
                    "",
                    "╭─────────────── Traceback (most recent call last) ────────────────╮\n"
                    "│ /pkg/colab_cli/commands/run.py:312 in run                        │\n"
                    "│   311     except TooManyAssignmentsError:                        │\n"
                    "│   312         typer.echo('Allocation refused')                   │\n"
                    "╰──────────────────────────────────────────────────────────────────╯\n"
                    "ApiError: HTTP 403 Forbidden\n",
                )
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_tb, sum_tb = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "rich-tb-403"),
            runner=rich_403_traceback_runner,
        )
        assert rc_tb == 1
        assert len(sum_tb["attempts"]) == 1
        assert sum_tb["attempts"][0]["tpu"] == "v6e1"

        # 14. Full ancestor chain validated for existing 0700 out_dir and explicit --config parent.
        gw_parent = tmp_path / "gw_parent"
        gw_parent.mkdir(mode=0o775)
        os.chmod(gw_parent, 0o775)
        nested_0700 = gw_parent / "private_0700"
        nested_0700.mkdir(mode=0o700)
        os.chmod(nested_0700, 0o700)
        rc_anc_out, sum_anc_out = run_benchmark(
            BenchConfig(dry_run=True, out_dir=nested_0700),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_anc_out == 2
        assert sum_anc_out["status"] == "untrusted_out_dir"

        rc_anc_cfg, sum_anc_cfg = run_benchmark(
            BenchConfig(
                dry_run=True,
                out_dir=tmp_path / "ok-out",
                config_path=nested_0700 / "sessions.json",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_anc_cfg == 2
        assert sum_anc_cfg["status"] == "untrusted_config_path"

        # 15. Another worker's concurrent session does NOT fail teardown when created_endpoint is gone;
        #     unknown created_endpoint + new endpoint is recorded as teardown_ambiguous.
        other_calls = {"n": 0}

        def other_worker_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                other_calls["n"] += 1
                if other_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[worker2-sess] m-s-otherworker99 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            return fallback_runner(cmd_list)

        rc_ow, sum_ow = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "other-worker"),
            runner=other_worker_runner,
        )
        assert rc_ow == 0
        assert sum_ow["status"] == "ok"
        assert sum_ow["teardown_verified"] is True
        assert sum_ow["other_new_endpoints"] == ["m-s-otherworker99"]

        # 16. `colab log` failure surfaces diagnostics and prevents silent `ok`, and probe script has no bare sys.exit(0).
        def log_fail_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "log" in cmd_list:
                return CommandResult(1, "", "colab log failed: export error")
            return fallback_runner(cmd_list)

        rc_lf, sum_lf = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "log-fail"),
            runner=log_fail_runner,
        )
        assert rc_lf == 1
        assert sum_lf["status"] == "proof_export_failed"
        assert sum_lf["proof_export"]["attempted"] is True
        assert sum_lf["proof_export"]["ok"] is False
        assert sum_lf["proof_export"]["returncode"] == 1
        assert "export error" in (sum_lf["proof_export"]["error"] or "")

        amb_calls = {"n": 0}

        def ambiguous_teardown_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                amb_calls["n"] += 1
                if amb_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[unknown] m-s-ambig77 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            return log_fail_runner(cmd_list)

        rc_amb, sum_amb = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "ambig-td"),
            runner=ambiguous_teardown_runner,
        )
        assert rc_amb == 1
        assert sum_amb["status"] == "teardown_ambiguous"
        assert sum_amb["ambiguous_endpoints"] == ["m-s-ambig77"]
        assert sum_amb["leaked_endpoints"] == []

        probe_src = default_tpu_probe_script()
        assert "if exit_rc != 0:\n    sys.exit(exit_rc)" in probe_src

        # 17. Auto-created out_dir validates parent/ancestors (rejects group-writable non-sticky TMPDIR).
        bad_tmpdir = tmp_path / "gw_tmp"
        bad_tmpdir.mkdir(mode=0o775)
        os.chmod(bad_tmpdir, 0o775)
        old_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(bad_tmpdir)
        try:
            rc_atmp, sum_atmp = run_benchmark(
                BenchConfig(dry_run=True, out_dir=None),
                runner=low_or_bad_cu_runner("unused"),
            )
            assert rc_atmp == 2
            assert sum_atmp["status"] == "untrusted_out_dir"
        finally:
            if old_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = old_tmpdir

        # 18. Symlinked lock files for explicit `--config` inside reused out_dir are rejected,
        #     and default `--config` paths are unique per run.
        reused_dir = tmp_path / "reused-default-cfg"
        reused_dir.mkdir(mode=0o700)
        os.chmod(reused_dir, 0o700)
        (reused_dir / "custom-sessions.json.lock").symlink_to(tmp_path / "outside.lock")
        rc_dlock, sum_dlock = run_benchmark(
            BenchConfig(
                dry_run=True,
                out_dir=reused_dir,
                config_path=reused_dir / "custom-sessions.json",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_dlock == 2
        assert sum_dlock["status"] == "untrusted_config_path"

        # 19. Atomic temp-file write+replace never mutates a hard-linked target.
        hl_dir = tmp_path / "hardlink-box"
        hl_dir.mkdir(mode=0o700)
        os.chmod(hl_dir, 0o700)
        hl_orig = hl_dir / "orig.txt"
        hl_orig.write_text("do-not-overwrite", encoding="utf-8")
        hl_copy = hl_dir / "summary.json"
        os.link(hl_orig, hl_copy)
        _write_text_nofollow(hl_copy, '{"ok": true}\n')
        assert hl_orig.read_text(encoding="utf-8") == "do-not-overwrite"
        assert hl_copy.read_text(encoding="utf-8") == '{"ok": true}\n'

        # 20. Real subprocess runner enforces explicit wall-timeout and sets timed_out=True.
        bounded = make_subprocess_runner(request_timeout_s=1.0, run_wall_timeout_s=0.2)
        res_to = bounded([sys.executable, "-c", "import time; time.sleep(2)", "run"])
        assert res_to.returncode == 124
        assert res_to.timed_out is True
        assert is_wall_timeout_error(res_to)
        assert is_allocation_timeout_error(
            res_to.returncode, res_to.stderr, timed_out=res_to.timed_out
        )

        # 21. Nonzero workload exit after `Session READY` still exports proof & preserves exit code.
        def ready_nonzero_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "run" in cmd_list:
                return CommandResult(
                    3,
                    "partial output\n",
                    "[colab] Session READY (m-s-nz99).\nTraceback: workload exited 3\n",
                )
            return fallback_runner(cmd_list)

        rc_nz, sum_nz = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "nonzero-ready"),
            runner=ready_nonzero_runner,
        )
        assert rc_nz == 3
        assert sum_nz["status"] == "failed"
        assert sum_nz["proof_export"]["attempted"] is True
        assert sum_nz["proof_export"]["ok"] is True
        assert sum_nz["created_endpoint"] == "tpu-v5e1-ep1"
        assert sum_nz["proof_notebook"] is not None
        assert Path(sum_nz["proof_notebook"]).is_file()

        # 22. Post-READY wall-clock timeout (rc=124) stops the active session using the same
        #     isolated --config BEFORE verifying release via `colab sessions`, and never falls back.
        wall_cmds: List[List[str]] = []
        wall_stopped = {"done": False}

        def post_ready_wall_timeout_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            wall_cmds.append(cmd_list)
            if "--help" in cmd_list or "version" in cmd_list or "usage" in cmd_list:
                res = base_help_usage_sessions(cmd_list)
                assert res is not None
                return res
            if "sessions" in cmd_list:
                # Before `stop`, the TPU would still be allocated if we failed to stop it!
                if any("run" in c for c in wall_cmds) and not wall_stopped["done"]:
                    return CommandResult(
                        0,
                        "[s] tpu-v6e1-wall1 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                        "",
                    )
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd_list:
                return CommandResult(
                    124,
                    "step 1...\nstep 2...\n",
                    "[colab] Session READY (wall-sess).\n"
                    "[colab_tpu_bench] `colab` subprocess exceeded wall timeout (480.0s); "
                    "stopping session and reconciling active sessions.\n",
                    timed_out=True,
                )
            if "stop" in cmd_list:
                wall_stopped["done"] = True
                return CommandResult(0, "[colab] Session terminated.\n", "")
            if "log" in cmd_list:
                out_idx = cmd_list.index("-o") + 1
                nb = {
                    "nbformat": 4,
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": "**Session Created**: now\n- Endpoint: `tpu-v6e1-wall1`\n",
                        }
                    ],
                }
                Path(cmd_list[out_idx]).write_text(json.dumps(nb), encoding="utf-8")
                return CommandResult(0, "Exported notebook.\n", "")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_wt, sum_wt = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "wall-timeout-post-ready"),
            runner=post_ready_wall_timeout_runner,
        )
        assert rc_wt == 124
        assert sum_wt["status"] == "wall_timeout"
        assert sum_wt["teardown_verified"] is True
        assert sum_wt["created_endpoint"] == "tpu-v6e1-wall1"
        assert sum_wt["proof_notebook"] is None
        assert sum_wt["metadata_notebook"] is not None
        assert sum_wt["proof_export"]["ok"] is False
        assert sum_wt["proof_export"]["metadata_only"] is True
        assert sum_wt["forced_stop"]["attempted"] is True
        assert sum_wt["forced_stop"]["returncode"] == 0
        assert sum_wt["forced_stop"]["error"] is None
        assert len(sum_wt["attempts"]) == 1
        assert sum_wt["attempts"][0]["tpu"] == "v6e1"
        stop_indices = [i for i, c in enumerate(wall_cmds) if "stop" in c]
        sess_indices = [i for i, c in enumerate(wall_cmds) if "sessions" in c]
        assert len(stop_indices) == 1
        assert len(sess_indices) == 2
        assert stop_indices[0] < sess_indices[1]
        stop_cmd = wall_cmds[stop_indices[0]]
        assert "--config" in stop_cmd and sum_wt["isolated_config"] in stop_cmd
        assert "-s" in stop_cmd and sum_wt["chosen_session"] in stop_cmd

        # 23. Pre-READY wall-clock timeout (rc=124) with no confirmed endpoint attempts
        #     `colab stop -s <sess>` without falling back, and keeps teardown_verified=False.
        pre_wall_cmds: List[List[str]] = []

        def pre_ready_wall_timeout_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            pre_wall_cmds.append(cmd_list)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(
                    124,
                    "",
                    "[colab] Creating session 'x'...\n"
                    "[colab_tpu_bench] `colab` subprocess exceeded wall timeout (480.0s); "
                    "stopping session and reconciling active sessions.\n",
                    timed_out=True,
                )
            if "stop" in cmd_list:
                return CommandResult(0, "[colab] Session 'x' not found.\n", "")
            if "log" in cmd_list:
                return CommandResult(1, "", "no log")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_pwt, sum_pwt = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "wall-timeout-pre-ready"),
            runner=pre_ready_wall_timeout_runner,
        )
        assert rc_pwt == 124
        assert sum_pwt["status"] == "allocation_timeout_unreconciled"
        assert sum_pwt["teardown_verified"] is False
        assert "in-flight" in (sum_pwt["teardown_error"] or "").lower()
        assert sum_pwt["forced_stop"]["attempted"] is True
        assert len(sum_pwt["attempts"]) == 1
        assert any("stop" in c for c in pre_wall_cmds)

        # 24. Post-run `colab sessions` exiting 0 with ADC auth failure on stderr (Colab CLI 0.8.0
        #     swallowed SystemExit when local store is empty) must NOT mark teardown_verified=True.
        adc_err_stderr = (
            "No valid default credentials found. To authenticate, run:\n\n"
            "  gcloud auth application-default login \\\n"
            "      --scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n"
        )
        adc_sess_calls = {"n": 0}

        def post_sessions_adc_fail_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                adc_sess_calls["n"] += 1
                if adc_sess_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[colab] No active sessions found on server.\n",
                    adc_err_stderr,
                )
            return fallback_runner(cmd_list)

        rc_adc_td, sum_adc_td = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "post-sessions-adc-fail"),
            runner=post_sessions_adc_fail_runner,
        )
        assert rc_adc_td == 1
        assert sum_adc_td["status"] == "teardown_verification_failed"
        assert sum_adc_td["teardown_verified"] is False
        assert "authentication error" in (sum_adc_td["teardown_error"] or "").lower()

        # Also after an uncertain allocation ReadTimeout -> allocation_timeout_unreconciled.
        adc_to_calls = {"n": 0}

        def alloc_timeout_adc_fail_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                adc_to_calls["n"] += 1
                if adc_to_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[colab] No active sessions found on server.\n",
                    adc_err_stderr,
                )
            return timeout_runner(cmd_list)

        rc_adc_to, sum_adc_to = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "alloc-to-adc-fail"),
            runner=alloc_timeout_adc_fail_runner,
        )
        assert rc_adc_to == 1
        assert sum_adc_to["status"] == "allocation_timeout_unreconciled"
        assert sum_adc_to["teardown_verified"] is False

        # 25. Pre-flight `colab sessions` exiting 0 with ADC auth failure aborts before `colab run`.
        pre_adc_cmds: List[List[str]] = []

        def pre_sessions_adc_fail_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            pre_adc_cmds.append(cmd_list)
            if "sessions" in cmd_list:
                return CommandResult(
                    0,
                    "[colab] No active sessions found on server.\n",
                    adc_err_stderr,
                )
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            raise AssertionError(f"Unexpected command after pre-flight sessions auth fail: {cmd_list}")

        rc_pre_adc, sum_pre_adc = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "pre-sessions-adc-fail"),
            runner=pre_sessions_adc_fail_runner,
        )
        assert rc_pre_adc == 1
        assert sum_pre_adc["status"] == "sessions_check_failed"
        assert not any("run" in c for c in pre_adc_cmds)

        # 26. Operator cancellation (KeyboardInterrupt) during `colab run` reconciles the TPU
        #     with `colab stop -s <sess>` + `colab sessions` and preserves exit code 130.
        cancel_cmds: List[List[str]] = []
        cancel_stopped = {"done": False}

        def cancel_clean_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            cancel_cmds.append(cmd_list)
            if "--help" in cmd_list or "version" in cmd_list or "usage" in cmd_list:
                res = base_help_usage_sessions(cmd_list)
                assert res is not None
                return res
            if "sessions" in cmd_list:
                if any("run" in c for c in cancel_cmds) and not cancel_stopped["done"]:
                    return CommandResult(
                        0,
                        "[s] tpu-v6e1-int1 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                        "",
                    )
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd_list:
                raise KeyboardInterrupt()
            if "stop" in cmd_list:
                cancel_stopped["done"] = True
                return CommandResult(0, "[colab] Session terminated.\n", "")
            if "log" in cmd_list:
                out_idx = cmd_list.index("-o") + 1
                nb = {
                    "nbformat": 4,
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": "**Session Created**: now\n- Endpoint: `tpu-v6e1-int1`\n",
                        }
                    ],
                }
                Path(cmd_list[out_idx]).write_text(json.dumps(nb), encoding="utf-8")
                return CommandResult(0, "Exported notebook.\n", "")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_int, sum_int = run_benchmark(
            BenchConfig(
                tpus=("v6e1", "v5e1"),
                out_dir=tmp_path / "cancel-clean",
                teardown_retry_delay_s=0.0,
            ),
            runner=cancel_clean_runner,
        )
        assert rc_int == 130
        assert sum_int["status"] == "interrupted"
        assert sum_int["teardown_verified"] is True
        assert sum_int["created_endpoint"] == "tpu-v6e1-int1"
        assert sum_int["proof_notebook"] is None
        assert sum_int["metadata_notebook"] is not None
        assert sum_int["proof_export"]["ok"] is False
        assert sum_int["proof_export"]["metadata_only"] is True
        assert sum_int["forced_stop"]["attempted"] is True
        assert sum_int["forced_stop"]["returncode"] == 0
        assert len(sum_int["attempts"]) == 1
        int_stop_idx = [i for i, c in enumerate(cancel_cmds) if "stop" in c][0]
        int_sess_idx = [i for i, c in enumerate(cancel_cmds) if "sessions" in c][1]
        assert int_stop_idx < int_sess_idx
        assert "--config" in cancel_cmds[int_stop_idx]
        assert sum_int["isolated_config"] in cancel_cmds[int_stop_idx]
        assert sum_int["chosen_session"] in cancel_cmds[int_stop_idx]

        # Unreconciled cancellation (stop fails and endpoint remains active) preserves rc=130
        # and keeps the recovery mapping in `isolated_config`.
        def cancel_unreconciled_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "stop" in cmd_list:
                return CommandResult(1, "", "stop failed: network error")
            if "sessions" in cmd_list and any("run" in c for c in cancel_unrec_cmds):
                return CommandResult(
                    0,
                    "[?] tpu-v6e1-int1 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            cancel_unrec_cmds.append(cmd_list)
            return cancel_clean_runner(cmd_list)

        cancel_unrec_cmds: List[List[str]] = []
        rc_intu, sum_intu = run_benchmark(
            BenchConfig(
                tpus=("v6e1",),
                out_dir=tmp_path / "cancel-unreconciled",
                teardown_retries=1,
                teardown_retry_delay_s=0.0,
            ),
            runner=cancel_unreconciled_runner,
        )
        assert rc_intu == 130
        assert sum_intu["status"] == "interrupted_unreconciled"
        assert sum_intu["teardown_verified"] is False
        assert "tpu-v6e1-int1" in sum_intu["leaked_endpoints"]
        assert sum_intu["recovery_command"] is not None
        assert sum_intu["recovery_config"] is not None
        unrec_cfg = json.loads(Path(sum_intu["recovery_config"]).read_text(encoding="utf-8"))
        assert unrec_cfg[sum_intu["chosen_session"]]["endpoint"] == "tpu-v6e1-int1"

        # 27. Transient `colab run` unassign failure that cleared local state is recovered by
        #     re-seeding `isolated_config`, retrying `colab stop -s <sess>`, and re-verifying.
        retry_sess_calls = {"n": 0}
        retry_released = {"done": False}
        retry_saw_seeded_endpoint: List[Optional[str]] = []

        def transient_leak_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                retry_sess_calls["n"] += 1
                if retry_sess_calls["n"] == 1 or retry_released["done"]:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[?] tpu-v5e1-ep1 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            if "stop" in cmd_list:
                cfg_idx = cmd_list.index("--config") + 1
                sess_idx = cmd_list.index("-s") + 1
                cfg_data = json.loads(Path(cmd_list[cfg_idx]).read_text(encoding="utf-8"))
                entry = cfg_data.get(cmd_list[sess_idx], {})
                retry_saw_seeded_endpoint.append(entry.get("endpoint"))
                assert entry.get("token_expires_at") is None
                retry_released["done"] = True
                return CommandResult(0, "[colab] Session terminated.\n", "")
            return fallback_runner(cmd_list)

        rc_tr, sum_tr = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                out_dir=tmp_path / "transient-leak-recovery",
                teardown_retries=2,
                teardown_retry_delay_s=0.0,
            ),
            runner=transient_leak_runner,
        )
        assert rc_tr == 0
        assert sum_tr["status"] == "ok"
        assert sum_tr["teardown_verified"] is True
        assert sum_tr["leaked_endpoints"] == []
        assert retry_saw_seeded_endpoint == ["tpu-v5e1-ep1"]
        assert len(sum_tr["release_retries"]) == 1
        assert sum_tr["release_retries"][0]["recovery_config"] is not None
        assert sum_tr["release_retries"][0]["recovery_config"] != sum_tr["isolated_config"]
        assert sum_tr["release_retries"][0]["returncode"] == 0

        # 28. `--wall-timeout` is independent of `--timeout` (unset disables local wall cap).
        unbounded_run = make_subprocess_runner(request_timeout_s=1.0, run_wall_timeout_s=None)
        res_unb = unbounded_run([sys.executable, "-c", "import time; time.sleep(0.25)", "run"])
        assert res_unb.returncode == 0
        assert res_unb.timed_out is False

        parsed_cli = build_parser().parse_args(["--timeout", "0", "--wall-timeout", "900"])
        assert parsed_cli.timeout == 0.0
        assert parsed_cli.wall_timeout == 900.0

        rc_bad_wt, sum_bad_wt = run_benchmark(
            BenchConfig(dry_run=True, wall_timeout_s=-5.0, out_dir=tmp_path / "bad-wall-to"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_bad_wt == 2
        assert sum_bad_wt["status"] == "invalid_timeout"

        # 29. Workload exiting 124 after normal teardown (`timed_out=False`) is NOT misclassified
        #     as a local `wall_timeout` and does NOT invoke a redundant `colab stop`.
        exit124_cmds: List[List[str]] = []

        def workload_exit_124_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            exit124_cmds.append(cmd_list)
            if "run" in cmd_list:
                return CommandResult(
                    124,
                    "custom timeout helper returned 124\n",
                    "[colab] Session READY (m-s-124).\n[colab] Session terminated.\n",
                    timed_out=False,
                )
            return fallback_runner(cmd_list)

        rc_124, sum_124 = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "workload-exit-124"),
            runner=workload_exit_124_runner,
        )
        assert rc_124 == 124
        assert sum_124["status"] == "failed"
        assert sum_124["teardown_verified"] is True
        assert sum_124["forced_stop"]["attempted"] is False
        assert not any("stop" in c for c in exit124_cmds)

        # 30. Pre-READY cancellation with no confirmed endpoint keeps teardown_verified=False
        #     even when the first `colab sessions` snapshot returns empty.
        def pre_ready_cancel_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                raise KeyboardInterrupt()
            if "stop" in cmd_list:
                return CommandResult(0, "[colab] Session not found.\n", "")
            if "log" in cmd_list:
                return CommandResult(1, "", "no log")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_prc, sum_prc = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "pre-ready-cancel"),
            runner=pre_ready_cancel_runner,
        )
        assert rc_prc == 130
        assert sum_prc["status"] == "interrupted_unreconciled"
        assert sum_prc["teardown_verified"] is False
        assert "in-flight" in (sum_prc["teardown_error"] or "").lower()

        # 31. SIGTERM reconciliation via real signal handler & subprocess runner preserves
        #     partial stdout/stderr already emitted before SIGTERM, stops the TPU, and exits 143.
        stub_dir = tmp_path / "stub-bin"
        stub_dir.mkdir(mode=0o700)
        os.chmod(stub_dir, 0o700)
        stub_marker = tmp_path / "stub-stopped.flag"
        stub_colab = stub_dir / "fake_colab.py"
        stub_colab.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, os, pathlib, sys, time
                args = sys.argv[1:]
                flag = pathlib.Path({str(stub_marker)!r})
                if "--help" in args:
                    print("Usage: colab --auth --config --no-public-api")
                    sys.exit(0)
                if "version" in args:
                    print("Version: 0.8.0")
                    sys.exit(0)
                if "usage" in args:
                    print("Current balance: 500.00 compute units")
                    print("Usage rate: 0.00/hr")
                    print("Active assignments: 0")
                    sys.exit(0)
                if "sessions" in args:
                    if flag.exists() or not (flag.parent / "stub-started.flag").exists():
                        print("[colab] No active sessions found on server.")
                    else:
                        print("[s] tpu-v6e1-sigterm | Hardware: V6E1 | Shape: STANDARD | Variant: TPU")
                    sys.exit(0)
                if "run" in args:
                    (flag.parent / "stub-started.flag").write_text("1", encoding="utf-8")
                    sys.stderr.write("[colab] Session READY (sigterm-sess).\\n")
                    sys.stderr.flush()
                    sys.stdout.write("PARTIAL_BENCH_STEP_1\\n")
                    sys.stdout.flush()
                    time.sleep(30)
                    sys.exit(0)
                if "stop" in args:
                    flag.write_text("stopped", encoding="utf-8")
                    print("[colab] Session terminated.")
                    sys.exit(0)
                if "log" in args:
                    out_p = pathlib.Path(args[args.index("-o") + 1])
                    nb = {{
                        "nbformat": 4,
                        "cells": [{{
                            "cell_type": "markdown",
                            "source": "**Session Created**: now\\n- Endpoint: `tpu-v6e1-sigterm`\\n",
                        }}],
                    }}
                    out_p.write_text(json.dumps(nb), encoding="utf-8")
                    print("Exported notebook.")
                    sys.exit(0)
                sys.exit(1)
                """
            ),
            encoding="utf-8",
        )
        os.chmod(stub_colab, 0o700)

        if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            started_flag = tmp_path / "stub-started.flag"

            def _send_sigterm_when_ready() -> None:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if started_flag.exists():
                        time.sleep(0.15)
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    time.sleep(0.02)

            killer = threading.Thread(target=_send_sigterm_when_ready, daemon=True)
            killer.start()
            sigterm_out = tmp_path / "sigterm-real"
            rc_term, sum_term = run_benchmark(
                BenchConfig(
                    colab_bin=str(stub_colab),
                    tpus=("v6e1", "v5e1"),
                    out_dir=sigterm_out,
                    teardown_retry_delay_s=0.0,
                )
            )
            killer.join(timeout=2.0)
            assert signal.getsignal(signal.SIGTERM) == prev_sigterm
            assert rc_term == 143
            assert sum_term["status"] == "terminated"
            assert sum_term["teardown_verified"] is True
            assert sum_term["forced_stop"]["attempted"] is True
            assert sum_term["forced_stop"]["returncode"] == 0
            assert stub_marker.exists()
            assert len(sum_term["attempts"]) == 1
            term_sess = sum_term["chosen_session"]
            assert term_sess is not None
            saved_stdout = (sigterm_out / f"{term_sess}.stdout.txt").read_text(encoding="utf-8")
            saved_stderr = (sigterm_out / f"{term_sess}.stderr.txt").read_text(encoding="utf-8")
            assert "PARTIAL_BENCH_STEP_1" in saved_stdout
            assert "[colab] Session READY" in saved_stderr
            assert "SIGTERM" in saved_stderr

        # 32. Intermediate symlink hopping through a world-writable scratch directory to a
        #     private 0700 directory MUST be rejected; trusted symlink chains canonicalize.
        priv_target = tmp_path / "priv-target"
        priv_target.mkdir(mode=0o700)
        os.chmod(priv_target, 0o700)
        scratch_dir = tmp_path / "scratch-ww"
        scratch_dir.mkdir(mode=0o777)
        os.chmod(scratch_dir, 0o777)
        scratch_hop = scratch_dir / "hop"
        scratch_hop.symlink_to(priv_target)
        trusted_parent = tmp_path / "trusted-parent"
        trusted_parent.mkdir(mode=0o700)
        os.chmod(trusted_parent, 0o700)
        outer_trap = trusted_parent / "entry"
        outer_trap.symlink_to(scratch_hop)

        rc_trap_out, sum_trap_out = run_benchmark(
            BenchConfig(dry_run=True, out_dir=outer_trap / "run-out"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_trap_out == 2
        assert sum_trap_out["status"] == "untrusted_out_dir"

        rc_trap_cfg, sum_trap_cfg = run_benchmark(
            BenchConfig(
                dry_run=True,
                out_dir=priv_target / "safe-out",
                config_path=outer_trap / "sessions.json",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_trap_cfg == 2
        assert sum_trap_cfg["status"] == "untrusted_config_path"

        safe_hop = trusted_parent / "safe-hop"
        safe_hop.symlink_to(priv_target)
        rc_safe_hop, sum_safe_hop = run_benchmark(
            BenchConfig(dry_run=True, out_dir=safe_hop / "canon-out"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_safe_hop == 0
        assert Path(sum_safe_hop["out_dir"]) == Path(os.path.realpath(priv_target / "canon-out"))
        assert Path(sum_safe_hop["isolated_config"]).parent == Path(sum_safe_hop["out_dir"])

        # 33. Mocked CommandCancelledError (SIGINT) preserves partial stdout/stderr into run logs.
        def sigint_partial_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "run" in cmd_list:
                raise CommandCancelledError(
                    signal_name="SIGINT",
                    returncode=130,
                    stdout="CTRLC_PROGRESS_OUTPUT\n",
                    stderr="[colab] Session READY (ctrlc-sess).\n",
                )
            return cancel_clean_runner(cmd_list)

        ctrlc_out = tmp_path / "ctrlc-partial"
        rc_cp, sum_cp = run_benchmark(
            BenchConfig(tpus=("v6e1",), out_dir=ctrlc_out, teardown_retry_delay_s=0.0),
            runner=sigint_partial_runner,
        )
        assert rc_cp == 130
        assert sum_cp["status"] == "interrupted"
        assert sum_cp["teardown_verified"] is True
        cp_sess = sum_cp["chosen_session"]
        assert cp_sess is not None
        assert "CTRLC_PROGRESS_OUTPUT" in (ctrlc_out / f"{cp_sess}.stdout.txt").read_text(encoding="utf-8")
        assert "[colab] Session READY" in (ctrlc_out / f"{cp_sess}.stderr.txt").read_text(encoding="utf-8")

        # 34. Colab CLI version gate requires >= 0.7.6 (final) before any authenticated command.
        assert is_supported_colab_cli_version("Version: 0.7.5") == (False, "0.7.5")
        assert is_supported_colab_cli_version("Version: 0.7.6rc1") == (False, "0.7.6rc1")
        assert is_supported_colab_cli_version("Version: 0.7.6.dev0") == (False, "0.7.6.dev0")
        assert is_supported_colab_cli_version("Version: 0.7.6") == (True, "0.7.6")
        assert is_supported_colab_cli_version("Version: 0.7.6.post1") == (True, "0.7.6.post1")
        assert is_supported_colab_cli_version("Version: 0.8.0") == (True, "0.8.0")
        assert is_supported_colab_cli_version("unknown") == (False, None)

        old_cli_cmds: List[List[str]] = []

        def old_cli_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            old_cli_cmds.append(cmd_list)
            if "--help" in cmd_list:
                return CommandResult(0, "Usage: colab --auth --config\n", "")
            if "version" in cmd_list:
                return CommandResult(0, "Version: 0.7.5\n", "")
            raise AssertionError(f"Authenticated command must not run on v0.7.5: {cmd_list}")

        rc_old, sum_old = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "old-cli-075"),
            runner=old_cli_runner,
        )
        assert rc_old == 2
        assert sum_old["status"] == "unsupported_colab_cli_version"
        assert sum_old["detected_version"] == "0.7.5"
        assert not any(sub in c for c in old_cli_cmds for sub in ("usage", "whoami", "sessions", "run"))

        # 35. Transient post-run `colab sessions` listing outage enters bounded retry loop,
        #     attempts `colab stop` using dedicated recovery config, and recovers (or preserves config).
        outage_sess_calls = {"n": 0}
        outage_stop_calls = {"n": 0}

        def transient_sessions_outage_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                outage_sess_calls["n"] += 1
                if outage_sess_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                if outage_sess_calls["n"] == 2:
                    return CommandResult(1, "", "HTTP 503 Service Unavailable")
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "stop" in cmd_list:
                outage_stop_calls["n"] += 1
                return CommandResult(0, "[colab] Session terminated.\n", "")
            return fallback_runner(cmd_list)

        rc_outage_ok, sum_outage_ok = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                out_dir=tmp_path / "outage-recover",
                teardown_retries=2,
                teardown_retry_delay_s=0.0,
            ),
            runner=transient_sessions_outage_runner,
        )
        assert rc_outage_ok == 0
        assert sum_outage_ok["status"] == "ok"
        assert sum_outage_ok["teardown_verified"] is True
        assert outage_sess_calls["n"] == 3
        assert outage_stop_calls["n"] == 1
        assert len(sum_outage_ok["release_retries"]) == 1

        # Persistent listing outage preserves per-run recovery config & recovery_command.
        perm_outage_calls = {"sess": 0, "stop": 0}

        def persistent_sessions_outage_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                perm_outage_calls["sess"] += 1
                if perm_outage_calls["sess"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(1, "", "HTTP 503 Service Unavailable")
            if "stop" in cmd_list:
                perm_outage_calls["stop"] += 1
                return CommandResult(1, "", "HTTP 503 Service Unavailable")
            return fallback_runner(cmd_list)

        rc_pout, sum_pout = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                out_dir=tmp_path / "outage-persist",
                teardown_retries=2,
                teardown_retry_delay_s=0.0,
            ),
            runner=persistent_sessions_outage_runner,
        )
        assert rc_pout == 1
        assert sum_pout["status"] == "teardown_verification_failed"
        assert sum_pout["teardown_verified"] is False
        assert perm_outage_calls["sess"] == 4
        assert perm_outage_calls["stop"] == 2
        assert sum_pout["recovery_config"] is not None
        assert sum_pout["recovery_command"] is not None
        assert Path(sum_pout["recovery_config"]).is_file()

        # 36. Recovery writes NEVER mutate an explicit shared `--config` file, and default
        #     per-run configs in a shared `--out-dir` are distinct.
        shared_cfg_dir = tmp_path / "shared-cfg-dir"
        shared_cfg_dir.mkdir(mode=0o700)
        os.chmod(shared_cfg_dir, 0o700)
        shared_cfg_file = shared_cfg_dir / "shared-sessions.json"
        other_worker_json = json.dumps(
            {
                "other-worker": {
                    "name": "other-worker",
                    "token": "tok",
                    "url": "https://example.invalid",
                    "endpoint": "m-s-other-worker",
                }
            },
            indent=2,
        )
        shared_cfg_file.write_text(other_worker_json, encoding="utf-8")
        os.chmod(shared_cfg_file, 0o600)

        rc_conc, sum_conc = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                out_dir=shared_cfg_dir,
                config_path=shared_cfg_file,
                teardown_retries=1,
                teardown_retry_delay_s=0.0,
            ),
            runner=leaked_runner,
        )
        assert rc_conc == 1
        assert shared_cfg_file.read_text(encoding="utf-8") == other_worker_json
        assert sum_conc["recovery_config"] != str(shared_cfg_file)
        assert Path(sum_conc["recovery_config"]).is_file()

        rc_d1, sum_d1 = run_benchmark(
            BenchConfig(dry_run=True, out_dir=shared_cfg_dir),
            runner=low_or_bad_cu_runner("unused"),
        )
        rc_d2, sum_d2 = run_benchmark(
            BenchConfig(dry_run=True, out_dir=shared_cfg_dir),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_d1 == 0 and rc_d2 == 0
        assert sum_d1["isolated_config"] != sum_d2["isolated_config"]

        # 37. Non-finite numeric values (nan / inf / -inf) are rejected before contacting Colab.
        nonfinite_cmds: List[List[str]] = []

        def forbid_calls_runner(cmd: Sequence[str]) -> CommandResult:
            nonfinite_cmds.append(list(cmd))
            raise AssertionError(f"Should not invoke CLI for non-finite config: {cmd}")

        for bad_cfg, expected_status in (
            (BenchConfig(min_cu_balance=float("nan"), out_dir=tmp_path / "nf1"), "invalid_cu_balance"),
            (BenchConfig(min_cu_balance=float("inf"), out_dir=tmp_path / "nf2"), "invalid_cu_balance"),
            (BenchConfig(timeout_s=float("nan"), out_dir=tmp_path / "nf3"), "invalid_timeout"),
            (BenchConfig(timeout_s=float("inf"), out_dir=tmp_path / "nf4"), "invalid_timeout"),
            (BenchConfig(wall_timeout_s=float("nan"), out_dir=tmp_path / "nf5"), "invalid_timeout"),
            (BenchConfig(wall_timeout_s=float("inf"), out_dir=tmp_path / "nf6"), "invalid_timeout"),
            (BenchConfig(request_timeout_s=float("nan"), out_dir=tmp_path / "nf7"), "invalid_timeout"),
            (BenchConfig(teardown_retry_delay_s=float("nan"), out_dir=tmp_path / "nf8"), "invalid_timeout"),
        ):
            rc_nf, sum_nf = run_benchmark(bad_cfg, runner=forbid_calls_runner)
            assert rc_nf == 2
            assert sum_nf["status"] == expected_status
        assert nonfinite_cmds == []

        # 38. `--dry-run` writes `dry-run-plan.json` and NEVER overwrites an existing `summary.json`.
        live_out = tmp_path / "live-preserved"
        live_out.mkdir(mode=0o700)
        os.chmod(live_out, 0o700)
        live_summary_content = '{"status": "ok", "chosen_tpu": "v6e1"}\n'
        (live_out / "summary.json").write_text(live_summary_content, encoding="utf-8")
        rc_drp, sum_drp = run_benchmark(
            BenchConfig(dry_run=True, out_dir=live_out),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_drp == 0
        assert (live_out / "summary.json").read_text(encoding="utf-8") == live_summary_content
        assert Path(sum_drp["plan_path"]).name == "dry-run-plan.json"
        assert Path(sum_drp["plan_path"]).is_file()

        # 39. SIGTERM arriving DURING teardown/verification (after `colab run` returns) stays
        #     caught until bounded stop retries, recovery config, and summary.json finish.
        if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
            prev_td_sig = signal.getsignal(signal.SIGTERM)
            td_sig_state = {"sess": 0, "stop": 0, "sent_sigterm": False}

            def sigterm_during_teardown_runner(cmd: Sequence[str]) -> CommandResult:
                cmd_list = list(cmd)
                if "sessions" in cmd_list:
                    td_sig_state["sess"] += 1
                    if td_sig_state["sess"] == 1:
                        return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                    if td_sig_state["sess"] == 2:
                        # Deliver real SIGTERM while post-run `colab sessions` is executing!
                        td_sig_state["sent_sigterm"] = True
                        os.kill(os.getpid(), signal.SIGTERM)
                        return CommandResult(
                            0,
                            "[?] tpu-v5e1-ep1 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                            "",
                        )
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                if "stop" in cmd_list:
                    td_sig_state["stop"] += 1
                    return CommandResult(0, "[colab] Session terminated.\n", "")
                return fallback_runner(cmd_list)

            td_sig_out = tmp_path / "sigterm-during-teardown"
            rc_tdsig, sum_tdsig = run_benchmark(
                BenchConfig(
                    tpus=("v5e1",),
                    out_dir=td_sig_out,
                    teardown_retries=2,
                    teardown_retry_delay_s=5.0,
                ),
                runner=sigterm_during_teardown_runner,
            )
            assert signal.getsignal(signal.SIGTERM) == prev_td_sig
            assert td_sig_state["sent_sigterm"] is True
            assert td_sig_state["stop"] == 1
            assert rc_tdsig == 143
            assert sum_tdsig["status"] == "terminated"
            assert sum_tdsig["teardown_verified"] is True
            assert (td_sig_out / "summary.json").is_file()

        # 40. Read-only existing `--config` or `.lock` file is rejected BEFORE provisioning
        #     (works both for non-root OS permission checks and root/CAP_DAC_OVERRIDE containers).
        from unittest.mock import patch as _mock_patch

        def _kernel_denies_write(path: Path) -> bool:
            try:
                fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            except PermissionError:
                return True
            else:
                os.close(fd)
                return False

        ro_dir = tmp_path / "readonly-cfg-dir"
        ro_dir.mkdir(mode=0o700)
        os.chmod(ro_dir, 0o700)
        ro_cfg = ro_dir / "ro-sessions.json"
        ro_cfg.write_text("{}\n", encoding="utf-8")
        os.chmod(ro_cfg, 0o400)
        try:
            if _kernel_denies_write(ro_cfg):
                rc_ro, sum_ro = run_benchmark(
                    BenchConfig(
                        tpus=("v5e1",),
                        out_dir=tmp_path / "ro-out-1",
                        config_path=ro_cfg,
                    ),
                    runner=forbid_calls_runner,
                )
                assert rc_ro == 2
                assert sum_ro["status"] == "untrusted_config_path"
            else:
                with _mock_patch("os.open", side_effect=PermissionError("read-only")):
                    rc_ro, sum_ro = run_benchmark(
                        BenchConfig(
                            tpus=("v5e1",),
                            out_dir=tmp_path / "ro-out-1",
                            config_path=ro_cfg,
                        ),
                        runner=forbid_calls_runner,
                    )
                    assert rc_ro == 2
                    assert sum_ro["status"] == "untrusted_config_path"
        finally:
            os.chmod(ro_cfg, 0o600)

        ro_lock_cfg = ro_dir / "locked-sessions.json"
        ro_lock = Path(f"{ro_lock_cfg}.lock")
        ro_lock.write_text("", encoding="utf-8")
        os.chmod(ro_lock, 0o400)
        try:
            if _kernel_denies_write(ro_lock):
                rc_rol, sum_rol = run_benchmark(
                    BenchConfig(
                        tpus=("v5e1",),
                        out_dir=tmp_path / "ro-out-2",
                        config_path=ro_lock_cfg,
                    ),
                    runner=forbid_calls_runner,
                )
                assert rc_rol == 2
                assert sum_rol["status"] == "untrusted_config_path"
            else:
                with _mock_patch("os.open", side_effect=PermissionError("read-only lock")):
                    rc_rol, sum_rol = run_benchmark(
                        BenchConfig(
                            tpus=("v5e1",),
                            out_dir=tmp_path / "ro-out-2",
                            config_path=ro_lock_cfg,
                        ),
                        runner=forbid_calls_runner,
                    )
                    assert rc_rol == 2
                    assert sum_rol["status"] == "untrusted_config_path"
        finally:
            os.chmod(ro_lock, 0o600)

        # 41. `--config` colliding with generated artifacts (e.g. `<out-dir>/summary.json`) is
        #     rejected BEFORE writing any files, preserving existing session mappings.
        coll_dir = tmp_path / "collision-dir"
        coll_dir.mkdir(mode=0o700)
        os.chmod(coll_dir, 0o700)
        existing_sessions_in_summary = (
            '{"other-worker": {"name": "other-worker", "token": "t", '
            '"url": "https://example.invalid", "endpoint": "m-s-keep"}}\n'
        )
        (coll_dir / "summary.json").write_text(existing_sessions_in_summary, encoding="utf-8")
        os.chmod(coll_dir / "summary.json", 0o600)

        for coll_name in (
            "summary.json",
            "dry-run-plan.json",
            "colab_tpu_probe_payload.py",
            "colab-recovery.other.json",
        ):
            rc_col, sum_col = run_benchmark(
                BenchConfig(
                    dry_run=True,
                    out_dir=coll_dir,
                    config_path=coll_dir / coll_name,
                ),
                runner=forbid_calls_runner,
            )
            assert rc_col == 2
            assert sum_col["status"] == "config_artifact_collision"
        assert (coll_dir / "summary.json").read_text(encoding="utf-8") == existing_sessions_in_summary
        assert not (coll_dir / "colab_tpu_probe_payload.py").exists()
        assert not (coll_dir / "dry-run-plan.json").exists()

        # 42. Custom scripts with `from __future__ import annotations` are wrapped and run in
        #     the real `__main__` module namespace (pickle-safe); syntax errors fail pre-provision.
        custom_dir = tmp_path / "custom-scripts"
        custom_dir.mkdir(mode=0o700)
        os.chmod(custom_dir, 0o700)
        future_script = custom_dir / "future_bench.py"
        future_script.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                from __future__ import annotations
                import pickle
                import sys

                def make_box(val: int) -> Box:
                    return Box(val)

                class Box:
                    def __init__(self, v: int) -> None:
                        self.v = v

                assert __name__ == "__main__"
                import os as _os
                assert _os.path.basename(__file__) == "future_bench.py"
                assert _os.path.isfile(__file__)
                assert sys.argv == [__file__, "--iters", "3"]
                b = make_box(7)
                assert b.v == 7
                assert pickle.loads(pickle.dumps(b)).v == 7
                """
            ),
            encoding="utf-8",
        )
        rc_fut, sum_fut = run_benchmark(
            BenchConfig(
                dry_run=True,
                script_path=future_script,
                script_args=("--iters", "3"),
                out_dir=tmp_path / "future-out",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_fut == 0
        assert sum_fut["wrapped_custom_script"] is True
        wrapper_file = Path(sum_fut["script"])
        assert wrapper_file != future_script
        assert wrapper_file.is_file()
        # Simulate exact `colab run` v0.8.0 prelude + wrapped script in a real Python __main__!
        colab_prelude = (
            "import sys, warnings\n"
            f"sys.argv = [{wrapper_file.name!r}, '--iters', '3']\n"
            "__name__ = '__main__'\n"
            "warnings.filterwarnings('ignore', message='To exit: use')\n"
        )
        combined_payload = colab_prelude + wrapper_file.read_text(encoding="utf-8")
        proc_fut = subprocess.run(
            [sys.executable, "-c", combined_payload],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_fut.returncode == 0, f"Wrapped script failed: {proc_fut.stderr}"

        bad_syntax_script = custom_dir / "broken.py"
        bad_syntax_script.write_text("def broken(:\n    pass\n", encoding="utf-8")
        rc_bsyn, sum_bsyn = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                script_path=bad_syntax_script,
                out_dir=tmp_path / "broken-script-out",
            ),
            runner=forbid_calls_runner,
        )
        assert rc_bsyn == 2
        assert sum_bsyn["status"] == "invalid_script"
        assert "syntax error" in (sum_bsyn["error"] or "").lower()

        # 43. `--config` aliasing `--script` OR pointing to unrelated/non-session files is
        #     rejected BEFORE calling Colab (without truncating or leaking file contents).
        orig_future_src = future_script.read_text(encoding="utf-8")
        rc_alias, sum_alias = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                script_path=future_script,
                config_path=future_script,
                out_dir=tmp_path / "alias-script-out",
            ),
            runner=forbid_calls_runner,
        )
        assert rc_alias == 2
        assert sum_alias["status"] == "config_artifact_collision"
        assert future_script.read_text(encoding="utf-8") == orig_future_src

        non_session_py = custom_dir / "notes.txt"
        secret_sentinel = "DO_NOT_LEAK_THIS_CONTENT_999"
        non_session_py.write_text(f"plain text {secret_sentinel}\n", encoding="utf-8")
        rc_unrel_txt, sum_unrel_txt = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                config_path=non_session_py,
                out_dir=tmp_path / "unrel-txt-out",
            ),
            runner=forbid_calls_runner,
        )
        assert rc_unrel_txt == 2
        assert sum_unrel_txt["status"] == "untrusted_config_path"
        assert secret_sentinel not in json.dumps(sum_unrel_txt)

        non_session_json = custom_dir / "not-a-store.json"
        non_session_json.write_text('{"status": "ok", "chosen_tpu": "v6e1"}\n', encoding="utf-8")
        rc_unrel_js, sum_unrel_js = run_benchmark(
            BenchConfig(
                tpus=("v5e1",),
                config_path=non_session_json,
                out_dir=tmp_path / "unrel-json-out",
            ),
            runner=forbid_calls_runner,
        )
        assert rc_unrel_js == 2
        assert sum_unrel_js["status"] == "untrusted_config_path"

        # 44. Pre-READY HTTP 504 Gateway Timeout on allocation POST does NOT fall back to v5e1
        #     and keeps `teardown_verified=False` (`allocation_timeout_unreconciled`).
        def gateway_504_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(
                    1,
                    "",
                    "[colab] Creating session 'x'...\n"
                    "ColabRequestError: 504 Server Error: Gateway Timeout for url: "
                    "https://colab.research.google.com/tun/m/assign\n",
                )
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_504, sum_504 = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "alloc-504"),
            runner=gateway_504_runner,
        )
        assert rc_504 == 1
        assert len(sum_504["attempts"]) == 1
        assert sum_504["attempts"][0]["tpu"] == "v6e1"
        assert sum_504["status"] == "allocation_timeout_unreconciled"
        assert sum_504["teardown_verified"] is False
        assert is_allocation_timeout_error(1, "HTTP 502 Bad Gateway")
        assert is_allocation_timeout_error(1, "503 Server Error: Service Unavailable")
        assert not is_allocation_timeout_error(1, 'File "/pkg/run.py", line 504, in run')

        # 45. Metadata-only exported notebook (no workload code cell) on a clean rc=0 run
        #     still verifies teardown using its endpoint, but marks proof incomplete.
        def metadata_only_log_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "log" in cmd_list:
                out_idx = cmd_list.index("-o") + 1
                nb = {
                    "nbformat": 4,
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "source": "**Session Created**: now\n- Endpoint: `tpu-v5e1-ep1`\n",
                        }
                    ],
                }
                Path(cmd_list[out_idx]).write_text(json.dumps(nb), encoding="utf-8")
                return CommandResult(0, "Exported notebook.\n", "")
            return fallback_runner(cmd_list)

        rc_meta, sum_meta = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "metadata-only-proof"),
            runner=metadata_only_log_runner,
        )
        assert rc_meta == 1
        assert sum_meta["status"] == "proof_export_failed"
        assert sum_meta["teardown_verified"] is True
        assert sum_meta["created_endpoint"] == "tpu-v5e1-ep1"
        assert sum_meta["proof_notebook"] is None
        assert sum_meta["metadata_notebook"] is not None
        assert sum_meta["proof_export"]["ok"] is False
        assert sum_meta["proof_export"]["metadata_only"] is True

        # 46. Managed subprocess runner preserves same-request allocation reconciliation
        #     (defaults COLAB_CLI_ASSIGN_RETRIES=1 when unset; honors caller setting; never forces 0).
        old_retries_env = os.environ.get("COLAB_CLI_ASSIGN_RETRIES")
        try:
            os.environ.pop("COLAB_CLI_ASSIGN_RETRIES", None)
            env_runner_def = make_subprocess_runner(request_timeout_s=10.0)
            res_env_def = env_runner_def(
                [sys.executable, "-c", "import os; print(os.environ['COLAB_CLI_ASSIGN_RETRIES'])"]
            )
            assert res_env_def.returncode == 0
            assert res_env_def.stdout.strip() == "1"

            os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "3"
            env_runner = make_subprocess_runner(request_timeout_s=10.0)
            res_env = env_runner(
                [sys.executable, "-c", "import os; print(os.environ['COLAB_CLI_ASSIGN_RETRIES'])"]
            )
            assert res_env.returncode == 0
            assert res_env.stdout.strip() == "3"
        finally:
            if old_retries_env is None:
                os.environ.pop("COLAB_CLI_ASSIGN_RETRIES", None)
            else:
                os.environ["COLAB_CLI_ASSIGN_RETRIES"] = old_retries_env

        # 47. Full `SessionState` schema validation rejects malformed `token_expires_at`,
        #     `last_execution`, or field types before calling Colab, while accepting valid stores.
        schema_dir = tmp_path / "schema-check"
        schema_dir.mkdir(mode=0o700)
        os.chmod(schema_dir, 0o700)
        base_entry = {
            "name": "s1",
            "token": "tok",
            "url": "https://example.invalid",
            "endpoint": "m-s-ep1",
        }
        bad_entries = (
            {**base_entry, "token_expires_at": "not-a-timestamp"},
            {**base_entry, "token_expires_at": "2026-11-22"},
            {**base_entry, "token_expires_at": 12345},
            {**base_entry, "variant": None},
            {**base_entry, "kernel_id": 999},
            {**base_entry, "last_execution": ["only-one"]},
            {**base_entry, "last_execution": [1, None, "2026-11-22 10:00:00"]},
        )
        for idx_bad, bad_val in enumerate(bad_entries):
            bad_store = schema_dir / f"bad-{idx_bad}.json"
            bad_store.write_text(json.dumps({"s1": bad_val}), encoding="utf-8")
            os.chmod(bad_store, 0o600)
            rc_bs, sum_bs = run_benchmark(
                BenchConfig(
                    tpus=("v5e1",),
                    config_path=bad_store,
                    out_dir=tmp_path / f"bad-schema-out-{idx_bad}",
                ),
                runner=forbid_calls_runner,
            )
            assert rc_bs == 2, f"Expected schema rejection for case {idx_bad}: {sum_bs}"
            assert sum_bs["status"] == "untrusted_config_path"

        good_store = schema_dir / "good.json"
        good_store.write_text(
            json.dumps(
                {
                    "s1": {
                        **base_entry,
                        "token_expires_at": "2026-11-22T14:00:00Z",
                        "variant": "TPU",
                        "accelerator": "V5E1",
                        "machine_shape": "STANDARD",
                        "kernel_id": "k-1",
                        "session_id": None,
                        "last_execution": ["bench.py", None, "2026-11-22 14:00:00"],
                        "running": None,
                    }
                }
            ),
            encoding="utf-8",
        )
        os.chmod(good_store, 0o600)
        rc_gs, sum_gs = run_benchmark(
            BenchConfig(dry_run=True, config_path=good_store, out_dir=tmp_path / "good-schema-out"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_gs == 0
        assert sum_gs["status"] == "dry_run"

        # 48. Pre-READY connection drop (`ConnectionError` / `RemoteDisconnected`) does NOT
        #     fall back to v5e1 and keeps `teardown_verified=False` (`allocation_timeout_unreconciled`).
        def conn_drop_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            base_res = base_help_usage_sessions(cmd_list)
            if base_res is not None:
                return base_res
            if "run" in cmd_list:
                return CommandResult(
                    1,
                    "",
                    "[colab] Creating session 'x'...\n"
                    "requests.exceptions.ConnectionError: ('Connection aborted.', "
                    "RemoteDisconnected('Remote end closed connection without response'))\n",
                )
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_cd, sum_cd = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "alloc-conn-drop"),
            runner=conn_drop_runner,
        )
        assert rc_cd == 1
        assert len(sum_cd["attempts"]) == 1
        assert sum_cd["attempts"][0]["tpu"] == "v6e1"
        assert sum_cd["status"] == "allocation_timeout_unreconciled"
        assert sum_cd["teardown_verified"] is False
        assert is_allocation_timeout_error(1, "ConnectionResetError: Connection reset by peer")
        assert is_allocation_timeout_error(1, "SSLError: unexpected eof while reading")

        # 49. Ordinary custom Python scripts (without `from __future__`) ALSO get wrapped so
        #     `Path(__file__)` works cleanly under `colab run`'s prelude.
        plain_script = custom_dir / "plain_file_user.py"
        plain_script.write_text(
            textwrap.dedent(
                """\
                from pathlib import Path
                import sys
                assert Path(__file__).name == "plain_file_user.py"
                assert Path(__file__).is_file()
                assert __name__ == "__main__"
                assert sys.argv == [__file__, "--ok"]
                """
            ),
            encoding="utf-8",
        )
        rc_plain, sum_plain = run_benchmark(
            BenchConfig(
                dry_run=True,
                script_path=plain_script,
                script_args=("--ok",),
                out_dir=tmp_path / "plain-file-out",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_plain == 0
        assert sum_plain["wrapped_custom_script"] is True
        plain_wrapper = Path(sum_plain["script"])
        plain_prelude = (
            "import sys, warnings\n"
            f"sys.argv = [{plain_wrapper.name!r}, '--ok']\n"
            "__name__ = '__main__'\n"
            "warnings.filterwarnings('ignore', message='To exit: use')\n"
        )
        proc_plain = subprocess.run(
            [sys.executable, "-c", plain_prelude + plain_wrapper.read_text(encoding="utf-8")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_plain.returncode == 0, f"Plain wrapped script failed: {proc_plain.stderr}"

        # 50. Omitted `--wall-timeout` uses a finite default (bounding stalled kernel setup),
        #     while explicit `--wall-timeout 0` opts out cleanly.
        assert resolve_effective_wall_timeout(300.0, None) == 480.0
        assert resolve_effective_wall_timeout(0.0, None) == 480.0
        assert resolve_effective_wall_timeout(300.0, 0.0) is None
        assert resolve_effective_wall_timeout(300.0, 900.0) == 900.0

        rc_def_wt, sum_def_wt = run_benchmark(
            BenchConfig(dry_run=True, timeout_s=300.0, wall_timeout_s=None, out_dir=tmp_path / "def-wt"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_def_wt == 0
        assert sum_def_wt["effective_wall_timeout_s"] == 480.0

        rc_opt_wt, sum_opt_wt = run_benchmark(
            BenchConfig(dry_run=True, timeout_s=300.0, wall_timeout_s=0.0, out_dir=tmp_path / "opt-wt"),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_opt_wt == 0
        assert sum_opt_wt["effective_wall_timeout_s"] is None

        # 51. Explicit `--request-timeout` overrides inherited COLAB_CLI_REQUEST_TIMEOUT, while
        #     omitting `--request-timeout` preserves a valid inherited environment value.
        old_req_env = os.environ.get("COLAB_CLI_REQUEST_TIMEOUT")
        os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = "300"
        try:
            explicit_runner = make_subprocess_runner(request_timeout_s=10.0)
            res_exp = explicit_runner(
                [sys.executable, "-c", "import os; print(os.environ['COLAB_CLI_REQUEST_TIMEOUT'])"]
            )
            assert res_exp.returncode == 0
            assert res_exp.stdout.strip() == "10.0"

            inherited_runner = make_subprocess_runner(request_timeout_s=None)
            res_inh = inherited_runner(
                [sys.executable, "-c", "import os; print(os.environ['COLAB_CLI_REQUEST_TIMEOUT'])"]
            )
            assert res_inh.returncode == 0
            assert res_inh.stdout.strip() == "300"

            parsed_default_req = build_parser().parse_args([])
            assert parsed_default_req.request_timeout is None
            parsed_explicit_req = build_parser().parse_args(["--request-timeout", "10"])
            assert parsed_explicit_req.request_timeout == 10.0

            rc_inh_dry, sum_inh_dry = run_benchmark(
                BenchConfig(dry_run=True, request_timeout_s=None, out_dir=tmp_path / "inh-req"),
                runner=low_or_bad_cu_runner("unused"),
            )
            assert rc_inh_dry == 0
            assert sum_inh_dry["effective_request_timeout_s"] == 300.0

            os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = "nan"
            rc_bad_env, sum_bad_env = run_benchmark(
                BenchConfig(dry_run=True, request_timeout_s=None, out_dir=tmp_path / "bad-env-req"),
                runner=forbid_calls_runner,
            )
            assert rc_bad_env == 2
            assert sum_bad_env["status"] == "invalid_timeout"
        finally:
            if old_req_env is None:
                os.environ.pop("COLAB_CLI_REQUEST_TIMEOUT", None)
            else:
                os.environ["COLAB_CLI_REQUEST_TIMEOUT"] = old_req_env

        # 52. Reject group/world-writable `--config` or `.lock` files and unsafe stored session names
        #     (path separators, `..`, leading `.`, NUL) before invoking Colab CLI.
        safe_parent = tmp_path / "perm-check"
        safe_parent.mkdir(mode=0o700)
        for bad_mode, tag in ((0o660, "grp-writable"), (0o602, "world-writable")):
            bad_mode_cfg = safe_parent / f"{tag}.json"
            bad_mode_cfg.write_text("{}", encoding="utf-8")
            os.chmod(bad_mode_cfg, bad_mode)
            rc_perm, sum_perm = run_benchmark(
                BenchConfig(config_path=bad_mode_cfg, out_dir=tmp_path / f"out-{tag}"),
                runner=forbid_calls_runner,
            )
            assert rc_perm == 2, f"Expected mode {oct(bad_mode)} config rejection: {sum_perm}"
            assert sum_perm["status"] == "untrusted_config_path"

        lock_parent = tmp_path / "lock-perm-check"
        lock_parent.mkdir(mode=0o700)
        ok_cfg_bad_lock = lock_parent / "sessions.json"
        ok_cfg_bad_lock.write_text("{}", encoding="utf-8")
        os.chmod(ok_cfg_bad_lock, 0o600)
        bad_lock = Path(f"{ok_cfg_bad_lock}.lock")
        bad_lock.write_text("", encoding="utf-8")
        os.chmod(bad_lock, 0o660)
        rc_lock_perm, sum_lock_perm = run_benchmark(
            BenchConfig(config_path=ok_cfg_bad_lock, out_dir=tmp_path / "out-lock-perm"),
            runner=forbid_calls_runner,
        )
        assert rc_lock_perm == 2, f"Expected group-writable lock rejection: {sum_lock_perm}"
        assert sum_lock_perm["status"] == "untrusted_config_path"

        for idx_bad, unsafe_name in enumerate(
            ("../escape", "a/b", "a\\b", ".hidden", "foo..bar", "bad\x00name")
        ):
            unsafe_cfg = safe_parent / f"unsafe-name-{idx_bad}.json"
            unsafe_cfg.write_text(
                json.dumps(
                    {
                        unsafe_name: {
                            "name": unsafe_name,
                            "token": "tok",
                            "url": "https://example.invalid",
                            "endpoint": "m-s-unsafe",
                        }
                    }
                ),
                encoding="utf-8",
            )
            os.chmod(unsafe_cfg, 0o600)
            rc_uname, sum_uname = run_benchmark(
                BenchConfig(config_path=unsafe_cfg, out_dir=tmp_path / f"out-uname-{idx_bad}"),
                runner=forbid_calls_runner,
            )
            assert rc_uname == 2, f"Expected unsafe stored name {unsafe_name!r} rejected: {sum_uname}"
            assert sum_uname["status"] == "untrusted_config_path"

        # 53. Match Colab/Pydantic RFC-3339 datetime validation for `token_expires_at`
        #     (reject hour-only timestamps, whitespace padding, compact ISO, bad tz offsets).
        for bad_ts in (
            "2026-11-22T14",
            "2026-11-22T14Z",
            "2026-11-22T14+00:00",
            " 2026-11-22T14:30:00Z ",
            "2026-11-22T14:30:00+25:00",
            "2026-11-22T14:30:00+00:60",
            "2026-11-22T14:30:00+00",
            "20261122T143000Z",
            "2026-02-29T14:30:00Z",
            "2026-11-22T14:30:00Z\n",
        ):
            assert not _is_valid_iso_datetime(bad_ts), f"Should reject {bad_ts!r}"
        for good_ts in (
            "2026-11-22T14:30",
            "2026-11-22T14:30Z",
            "2026-11-22T14:30:00",
            "2026-11-22T14:30:00Z",
            "2026-11-22T14:30:00.123456Z",
            "2026-11-22T14:30:00.123456789Z",
            "2026-11-22 14:30:00-0500",
            "2024-02-29T14:30:00+05:30",
        ):
            assert _is_valid_iso_datetime(good_ts), f"Should accept {good_ts!r}"

        hour_only_cfg = safe_parent / "hour-only-ts.json"
        hour_only_cfg.write_text(
            json.dumps(
                {
                    "keep-me": {
                        "name": "keep-me",
                        "token": "tok",
                        "url": "https://example.invalid",
                        "endpoint": "m-s-keep",
                        "token_expires_at": "2026-11-22T14",
                    }
                }
            ),
            encoding="utf-8",
        )
        os.chmod(hour_only_cfg, 0o600)
        rc_hr, sum_hr = run_benchmark(
            BenchConfig(config_path=hour_only_cfg, out_dir=tmp_path / "out-hour-only"),
            runner=forbid_calls_runner,
        )
        assert rc_hr == 2, f"Expected hour-only timestamp rejected: {sum_hr}"
        assert sum_hr["status"] == "untrusted_config_path"

        # 54. Recover TPU endpoint independently from Colab CLI history JSONL even when
        #     notebook export fails (e.g., disk full) and local session mapping was removed,
        #     so `[?]` in `colab sessions` triggers stop retries instead of ambiguity.
        sample_probe: Dict[str, Any] = {
            "jax": {
                "version": "0.5.0",
                "default_backend": "tpu",
                "device_count": 1,
                "devices": [{"id": 0, "platform": "tpu"}],
            },
            "benchmark": {
                "hardware": "colab-tpu",
                "hw_us": 120.0,
                "gflops": 1000.0,
            },
        }
        fake_home = tmp_path / "hist-home"
        hist_dir = fake_home / ".config" / "colab-cli" / "history"
        hist_dir.mkdir(parents=True, mode=0o700)
        os.chmod(fake_home, 0o700)
        os.chmod(fake_home / ".config", 0o700)
        os.chmod(fake_home / ".config" / "colab-cli", 0o700)
        os.chmod(hist_dir, 0o700)
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(fake_home)
        try:
            hist_state = {"stopped": False, "stop_configs": [], "sessions_calls": 0}

            def history_recovery_runner(cmd: Sequence[str]) -> CommandResult:
                if "--help" in cmd:
                    return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
                if "version" in cmd:
                    return CommandResult(0, "Version: 0.8.0\n", "")
                if "usage" in cmd:
                    return CommandResult(
                        0,
                        "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                        "",
                    )
                if "run" in cmd:
                    s_idx = cmd.index("-s")
                    sess_name = cmd[s_idx + 1]
                    # Simulate Colab CLI 0.8.0 writing `session_created` to history JSONL and then
                    # removing the session mapping from `--config` even though unassign failed.
                    hfile = hist_dir / f"{sess_name}.jsonl"
                    hfile.write_text(
                        json.dumps(
                            {
                                "timestamp": "2026-11-22T15:30:00+00:00",
                                "event_type": "session_created",
                                "endpoint": "m-s-hist-recovered-99",
                                "accelerator": "V6E1",
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    os.chmod(hfile, 0o600)
                    return CommandResult(
                        0,
                        f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n",
                        f"[colab] Session READY ({sess_name}). Executing...\n",
                    )
                if "log" in cmd:
                    return CommandResult(1, "", "OSError: [Errno 28] No space left on device\n")
                if "sessions" in cmd:
                    hist_state["sessions_calls"] += 1
                    if hist_state["sessions_calls"] == 1 or hist_state["stopped"]:
                        return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                    # Active TPU appears as `[?]` because Colab CLI removed the local mapping!
                    return CommandResult(
                        0,
                        "[?] m-s-hist-recovered-99 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                        "",
                    )
                if "stop" in cmd:
                    c_idx = cmd.index("--config")
                    rec_cfg_path = Path(cmd[c_idx + 1])
                    rec_data = json.loads(rec_cfg_path.read_text(encoding="utf-8"))
                    s_idx = cmd.index("-s")
                    sess_name = cmd[s_idx + 1]
                    assert rec_data[sess_name]["endpoint"] == "m-s-hist-recovered-99"
                    hist_state["stop_configs"].append(str(rec_cfg_path))
                    hist_state["stopped"] = True
                    return CommandResult(0, "[colab] Stopped session.\n", "")
                raise AssertionError(f"Unexpected cmd: {cmd}")

            rc_hrec, sum_hrec = run_benchmark(
                BenchConfig(
                    tpus=("v6e1",),
                    out_dir=tmp_path / "hist-recovery-out",
                    teardown_retries=2,
                    teardown_retry_delay_s=0.0,
                ),
                runner=history_recovery_runner,
            )
            assert sum_hrec["created_endpoint"] == "m-s-hist-recovered-99", sum_hrec
            assert sum_hrec["teardown_verified"] is True, sum_hrec
            assert sum_hrec["ambiguous_endpoints"] == [], sum_hrec
            assert sum_hrec["leaked_endpoints"] == [], sum_hrec
            assert len(sum_hrec["release_retries"]) == 1, sum_hrec
            assert sum_hrec["release_retries"][0]["endpoint"] == "m-s-hist-recovered-99"
            assert sum_hrec["proof_export"]["ok"] is False
            assert sum_hrec["status"] == "proof_export_failed", sum_hrec
            assert rc_hrec == 1
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home

        # 55. Bound in-memory capture for verbose workloads while streaming full logs to private
        #     files, preserving probe JSON in the middle, and surviving disk write OSErrors.
        stream_dir = tmp_path / "stream-logs"
        stream_dir.mkdir(mode=0o700)
        os.chmod(stream_dir, 0o700)
        verbose_script = tmp_path / "emit_verbose.py"
        verbose_script.write_text(
            textwrap.dedent(
                f"""\
                import json, sys
                chunk = "A" * 65536 + "\\n"
                for _ in range(16):
                    sys.stdout.write(chunk)
                sys.stdout.write({PROBE_JSON_BEGIN!r} + "\\n")
                sys.stdout.write(json.dumps({sample_probe!r}) + "\\n")
                sys.stdout.write({PROBE_JSON_END!r} + "\\n")
                for _ in range(16):
                    sys.stdout.write("B" * 65536 + "\\n")
                sys.stderr.write("[colab] Session READY (verbose-sess). Executing...\\n")
                """
            ),
            encoding="utf-8",
        )
        bounded_runner = make_subprocess_runner(
            request_timeout_s=30.0,
            run_wall_timeout_s=30.0,
            log_dir=stream_dir,
            max_capture_bytes=8192,
        )
        # Simulate a `colab run -s verbose-sess` invocation using sys.executable so `-s` log
        # routing activates in `make_subprocess_runner`.
        res_verb = bounded_runner(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[-1], run_name='__main__')",
                "run",
                "-s",
                "verbose-sess",
                "--",
                str(verbose_script),
            ]
        )
        assert res_verb.returncode == 0, res_verb.stderr
        assert res_verb.stdout_log_written is True
        assert res_verb.log_write_error is None
        assert len(res_verb.stdout) < 64 * 1024, f"Expected bounded RAM, got {len(res_verb.stdout)}"
        parsed_verb = extract_probe_json(res_verb.stdout)
        assert parsed_verb == sample_probe, f"Probe lost in bounded stream: {parsed_verb}"
        streamed_log = stream_dir / "verbose-sess.stdout.txt"
        assert streamed_log.is_file()
        assert streamed_log.stat().st_size > 2 * 1024 * 1024
        assert (stat.S_IMODE(streamed_log.stat().st_mode) & 0o077) == 0

        # Verify disk-full OSError during log write degrades cleanly without crashing capture.
        cap_disk_full = _BoundedStreamCapture(
            log_path=stream_dir / "disk-full.stdout.txt",
            max_capture_bytes=2048,
            capture_probe=True,
        )
        class _FailWriter:
            def write(self, _data: bytes) -> int:
                raise OSError(28, "No space left on device")
            def close(self) -> None:
                pass
        cap_disk_full._fh = _FailWriter()
        cap_disk_full.feed(b"head-" * 300)
        cap_disk_full.feed(
            f"\n{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n".encode("utf-8")
        )
        cap_disk_full.feed(b"tail-" * 300)
        cap_disk_full.finalize(keep_log=True)
        assert cap_disk_full.write_error is not None
        assert cap_disk_full.log_written is False
        assert extract_probe_json(cap_disk_full.render_text()) == sample_probe

        # 56. Pre-READY HTTP 500 Internal Server Error keeps allocation outcome unresolved
        #     (blocks v5e1 fallback, keeps teardown_verified=False, surfaces recheck guidance).
        assert is_allocation_timeout_error(
            1,
            "requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://colab.research.google.com/tun/m/assign",
        )
        assert is_allocation_timeout_error(1, "HTTP Error 500: Internal Server Error")
        assert is_allocation_timeout_error(1, "status_code=500")
        assert not is_capacity_or_quota_error(
            1, "", "requests.exceptions.HTTPError: 500 Server Error: Internal Server Error"
        )

        http500_runs: List[str] = []

        def http500_alloc_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                tpu_arg = cmd[cmd.index("--tpu") + 1]
                http500_runs.append(tpu_arg)
                return CommandResult(
                    1,
                    "",
                    "requests.exceptions.HTTPError: 500 Server Error: Internal Server Error for url: https://colab.research.google.com/tun/m/assign\n",
                )
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_500, sum_500 = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "http500-out"),
            runner=http500_alloc_runner,
        )
        assert http500_runs == ["V6E1"], f"HTTP 500 must not fall back to V5E1: {http500_runs}"
        assert rc_500 != 0
        assert sum_500["teardown_verified"] is False, sum_500
        assert sum_500["status"] == "allocation_timeout_unreconciled", sum_500
        assert "sessions" in (sum_500["teardown_error"] or ""), sum_500

        # 57. Wrapped custom script materializes its source file at runtime so both
        #     `open(__file__)` and `inspect.getsource(fn)` succeed.
        inspect_script = custom_dir / "inspect_self_bench.py"
        inspect_script.write_text(
            textwrap.dedent(
                """\
                from __future__ import annotations
                import inspect
                from pathlib import Path

                def sample_kernel_fn(x: int) -> int:
                    return x * 2 + 1

                src_disk = Path(__file__).read_text(encoding="utf-8")
                assert "def sample_kernel_fn(x: int) -> int:" in src_disk
                fn_src = inspect.getsource(sample_kernel_fn)
                assert "return x * 2 + 1" in fn_src
                assert Path(__file__).name == "inspect_self_bench.py"
                """
            ),
            encoding="utf-8",
        )
        rc_insp, sum_insp = run_benchmark(
            BenchConfig(
                dry_run=True,
                script_path=inspect_script,
                out_dir=tmp_path / "inspect-self-out",
            ),
            runner=low_or_bad_cu_runner("unused"),
        )
        assert rc_insp == 0
        assert sum_insp["wrapped_custom_script"] is True
        insp_wrapper = Path(sum_insp["script"])
        insp_prelude = (
            "import sys, warnings\n"
            f"sys.argv = [{insp_wrapper.name!r}]\n"
            "__name__ = '__main__'\n"
            "warnings.filterwarnings('ignore', message='To exit: use')\n"
        )
        proc_insp = subprocess.run(
            [sys.executable, "-c", insp_prelude + insp_wrapper.read_text(encoding="utf-8")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_insp.returncode == 0, f"inspect/open(__file__) script failed: {proc_insp.stderr}"

        # 58. Same-request allocation reconciliation stays enabled while fallback is still blocked
        #     after an internal retry + 412, a request-timeout-duration 412, or a materialized
        #     assignment appearing in the pre-fallback `colab sessions` check.
        retry_412_stderr = (
            "WARNING: Assignment request failed; retrying in 10s\n"
            "[colab] Allocation refused (precondition failed): 412\n"
        )
        assert is_allocation_timeout_error(1, retry_412_stderr)
        assert not is_capacity_or_quota_error(1, "", retry_412_stderr)

        slow_412_runs: List[str] = []

        def slow_412_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                slow_412_runs.append(cmd[cmd.index("--tpu") + 1])
                return CommandResult(
                    1,
                    "",
                    "[colab] Allocation refused (precondition failed): 412\n",
                    elapsed_s=65.0,
                )
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_s412, sum_s412 = run_benchmark(
            BenchConfig(
                tpus=("v6e1", "v5e1"),
                request_timeout_s=60.0,
                out_dir=tmp_path / "slow-412-out",
            ),
            runner=slow_412_runner,
        )
        assert slow_412_runs == ["V6E1"], f"Slow 412 must block V5E1 fallback: {slow_412_runs}"
        assert rc_s412 != 0
        assert sum_s412["teardown_verified"] is False
        assert sum_s412["status"] == "allocation_timeout_unreconciled"

        # Pre-fallback `colab sessions` detecting a newly materialized endpoint aborts V5E1 fallback.
        prefb_runs: List[str] = []
        prefb_sessions_n = {"n": 0}

        def prefb_materialized_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "run" in cmd:
                prefb_runs.append(cmd[cmd.index("--tpu") + 1])
                return CommandResult(
                    1, "", "[colab] Allocation refused (precondition failed): 412\n", elapsed_s=5.0
                )
            if "sessions" in cmd:
                prefb_sessions_n["n"] += 1
                if prefb_sessions_n["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[?] m-s-late-materialized-1 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_pfb, sum_pfb = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "prefb-mat-out"),
            runner=prefb_materialized_runner,
        )
        assert prefb_runs == ["V6E1"], f"Materialized endpoint must abort fallback: {prefb_runs}"
        assert rc_pfb != 0
        assert sum_pfb["teardown_verified"] is False
        assert sum_pfb["ambiguous_endpoints"] == ["m-s-late-materialized-1"]

        # 59. When recovery-store seeding fails:
        #     (a) reuse the original `--config` if it still maps the session to the endpoint;
        #     (b) when neither recovery seeding nor original store has a mapping, skip calling
        #         `colab stop` on an unseeded config and report seed failure + manual guidance.
        orig_fallback_out = tmp_path / "seed-fail-orig-out"
        orig_fallback_out.mkdir(mode=0o700)
        os.chmod(orig_fallback_out, 0o700)
        orig_store_cfg = orig_fallback_out / "orig-store.json"
        orig_store_cfg.write_text("{}", encoding="utf-8")
        os.chmod(orig_store_cfg, 0o600)
        orig_stop_cmds: List[List[str]] = []
        orig_state = {"stopped": False, "sessions_calls": 0, "sess": ""}

        def seed_fail_orig_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "run" in cmd:
                s_name = cmd[cmd.index("-s") + 1]
                orig_state["sess"] = s_name
                orig_store_cfg.write_text(
                    json.dumps(
                        {
                            s_name: {
                                "name": s_name,
                                "token": "tok",
                                "url": "https://example.invalid",
                                "endpoint": "m-s-orig-mapped-1",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return CommandResult(
                    0,
                    f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n",
                    f"[colab] Session READY ({s_name}). Executing...\n",
                )
            if "log" in cmd:
                return CommandResult(1, "", "disk full")
            if "sessions" in cmd:
                orig_state["sessions_calls"] += 1
                if orig_state["sessions_calls"] == 1 or orig_state["stopped"]:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    f"[{orig_state['sess']}] m-s-orig-mapped-1 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            if "stop" in cmd:
                orig_stop_cmds.append(list(cmd))
                orig_state["stopped"] = True
                return CommandResult(0, "[colab] Stopped.\n", "")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        with _mock_patch(
            f"{__name__}._write_recovery_session_store",
            side_effect=OSError(28, "No space left on device"),
        ):
            rc_sfo, sum_sfo = run_benchmark(
                BenchConfig(
                    tpus=("v6e1",),
                    config_path=orig_store_cfg,
                    out_dir=orig_fallback_out,
                    teardown_retries=2,
                    teardown_retry_delay_s=0.0,
                ),
                runner=seed_fail_orig_runner,
            )
        assert len(orig_stop_cmds) == 1, orig_stop_cmds
        assert str(orig_store_cfg.resolve()) in orig_stop_cmds[0], orig_stop_cmds[0]
        assert sum_sfo["teardown_verified"] is True, sum_sfo

        # (b) No recovery store AND original store has no mapping: never call `colab stop` with
        #     an unseeded config and never advertise an unusable recovery_command.
        noseed_stop_cmds: List[List[str]] = []
        noseed_sessions_calls = {"n": 0}
        fake_home2 = tmp_path / "noseed-home"
        hist_dir2 = fake_home2 / ".config" / "colab-cli" / "history"
        hist_dir2.mkdir(parents=True, mode=0o700)
        os.chmod(fake_home2, 0o700)
        os.chmod(fake_home2 / ".config", 0o700)
        os.chmod(fake_home2 / ".config" / "colab-cli", 0o700)
        os.chmod(hist_dir2, 0o700)
        old_home2 = os.environ.get("HOME")
        os.environ["HOME"] = str(fake_home2)
        try:
            def seed_fail_unmapped_runner(cmd: Sequence[str]) -> CommandResult:
                if "--help" in cmd:
                    return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
                if "version" in cmd:
                    return CommandResult(0, "Version: 0.8.0\n", "")
                if "usage" in cmd:
                    return CommandResult(
                        0,
                        "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                        "",
                    )
                if "run" in cmd:
                    s_name = cmd[cmd.index("-s") + 1]
                    hf = hist_dir2 / f"{s_name}.jsonl"
                    hf.write_text(
                        json.dumps({"event_type": "session_created", "endpoint": "m-s-unmapped-77"}) + "\n",
                        encoding="utf-8",
                    )
                    os.chmod(hf, 0o600)
                    return CommandResult(
                        0,
                        f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n",
                        f"[colab] Session READY ({s_name}). Executing...\n",
                    )
                if "log" in cmd:
                    return CommandResult(1, "", "disk full")
                if "sessions" in cmd:
                    noseed_sessions_calls["n"] += 1
                    if noseed_sessions_calls["n"] == 1:
                        return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                    return CommandResult(
                        0,
                        "[?] m-s-unmapped-77 | Hardware: V6E1 | Shape: STANDARD | Variant: TPU\n",
                        "",
                    )
                if "stop" in cmd:
                    noseed_stop_cmds.append(list(cmd))
                    return CommandResult(1, "", "unseeded")
                raise AssertionError(f"Unexpected cmd: {cmd}")

            with _mock_patch(
                f"{__name__}._write_recovery_session_store",
                side_effect=OSError(28, "No space left on device"),
            ):
                rc_ns, sum_ns = run_benchmark(
                    BenchConfig(
                        tpus=("v6e1",),
                        out_dir=tmp_path / "noseed-out",
                        teardown_retries=1,
                        teardown_retry_delay_s=0.0,
                    ),
                    runner=seed_fail_unmapped_runner,
                )
            assert noseed_stop_cmds == [], f"Must not call colab stop with unseeded config: {noseed_stop_cmds}"
            assert rc_ns != 0
            assert sum_ns["teardown_verified"] is False
            assert sum_ns["recovery_command"] is None
            assert sum_ns["recovery_config"] is None
            assert "m-s-unmapped-77" in (sum_ns["teardown_error"] or "")
            assert "Manage Sessions" in (sum_ns["teardown_error"] or "")
        finally:
            if old_home2 is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home2

        # 60. Failed or incomplete `colab --help` must NEVER drop `--auth` or `--config`.
        help_fail_seen: List[List[str]] = []

        def broken_help_runner(cmd: Sequence[str]) -> CommandResult:
            help_fail_seen.append(list(cmd))
            if "--help" in cmd:
                return CommandResult(1, "", "transient help error")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 1.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_bh, sum_bh = run_benchmark(
            BenchConfig(auth="adc", min_cu_balance=5.0, out_dir=tmp_path / "broken-help-out"),
            runner=broken_help_runner,
        )
        assert rc_bh == 1
        assert sum_bh["status"] == "cu_budget_insufficient"
        usage_cmds_bh = [c for c in help_fail_seen if "usage" in c]
        assert len(usage_cmds_bh) == 1
        assert "--auth" in usage_cmds_bh[0] and "adc" in usage_cmds_bh[0]
        assert "--config" in usage_cmds_bh[0]

        # 61. Repeated runs sharing an output directory preserve separate `<session>.summary.json`
        #     files alongside `summary.json`.
        shared_out = tmp_path / "shared-summary-out"

        def ok_summary_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                s_name = cmd[cmd.index("-s") + 1]
                return CommandResult(
                    0,
                    f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n",
                    f"[colab] Session READY ({s_name}). Executing...\n",
                )
            if "log" in cmd:
                nb_out = Path(cmd[cmd.index("-o") + 1])
                nb_out.write_text(
                    json.dumps(
                        {
                            "nbformat": 4,
                            "cells": [
                                {"cell_type": "markdown", "source": "- Endpoint: `m-s-shared-ep`"},
                                {"cell_type": "code", "source": "print('ok')"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return CommandResult(0, "Exported.\n", "")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_sh1, sum_sh1 = run_benchmark(
            BenchConfig(tpus=("v6e1",), out_dir=shared_out), runner=ok_summary_runner
        )
        rc_sh2, sum_sh2 = run_benchmark(
            BenchConfig(tpus=("v6e1",), out_dir=shared_out), runner=ok_summary_runner
        )
        assert rc_sh1 == 0 and rc_sh2 == 0
        p_sum1 = Path(sum_sh1["run_summary"])
        p_sum2 = Path(sum_sh2["run_summary"])
        assert p_sum1 != p_sum2
        assert p_sum1.is_file() and p_sum2.is_file()
        assert json.loads(p_sum1.read_text(encoding="utf-8"))["chosen_session"] == sum_sh1["chosen_session"]
        assert json.loads(p_sum2.read_text(encoding="utf-8"))["chosen_session"] == sum_sh2["chosen_session"]
        assert (shared_out / "summary.json").is_file()

        # 62. A partial opening probe marker in the truncated head buffer must NOT mask the
        #     complete probe payload captured separately and appended after the omission banner.
        cap_partial_head = _BoundedStreamCapture(
            max_capture_bytes=512,
            capture_probe=True,
        )
        # Place PROBE_JSON_BEGIN right near byte 240 so the 256-byte head retains BEGIN + partial
        # JSON but cuts off before END, followed by > 2 KB of trailing output.
        probe_block = f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n"
        cap_partial_head.feed(b"X" * 210 + probe_block.encode("utf-8") + b"Y" * 3000)
        rendered_ph = cap_partial_head.render_text()
        assert rendered_ph.count(PROBE_JSON_BEGIN) == 2
        assert cap_partial_head.captured_probe_text() is not None
        assert extract_probe_json(cap_partial_head.captured_probe_text() or "") == sample_probe
        assert extract_probe_json(rendered_ph) == sample_probe

        # 63. Capture internal allocation retry / 5xx signals via `--logtostderr` on `colab run`
        #     so fast 502/503/504 + retry + 412 blocks fallback AND keeps teardown unverified
        #     even with `--no-fallback`; also treat 412 as inconclusive if `--logtostderr` is absent.
        fast_502_412_cmds: List[List[str]] = []

        def fast_502_412_runner(cmd: Sequence[str]) -> CommandResult:
            fast_502_412_cmds.append(list(cmd))
            if "--help" in cmd:
                return CommandResult(
                    0, "Usage: colab --auth --config --logtostderr --no-public-api\n", ""
                )
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                assert "--logtostderr" in cmd, f"Expected --logtostderr on run cmd: {cmd}"
                return CommandResult(
                    1,
                    "",
                    (
                        "2026-11-22 16:00:00,000 - colab_cli.client - DEBUG - Response: 502 Bad Gateway\n"
                        "2026-11-22 16:00:00,001 - colab_cli.client - WARNING - Assignment request failed; retrying in 10s\n"
                        "2026-11-22 16:00:10,200 - colab_cli.client - DEBUG - Response: 412 Precondition Failed\n"
                        "[colab] Allocation refused (precondition failed).\n"
                    ),
                    elapsed_s=11.0,
                )
            raise AssertionError(f"Unexpected cmd: {cmd}")

        # With fallback enabled (v6e1 -> v5e1): must NOT fall back to v5e1 and must NOT verify teardown.
        rc_f502_fb, sum_f502_fb = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "fast-502-412-fb"),
            runner=fast_502_412_runner,
        )
        run_cmds_502_fb = [c for c in fast_502_412_cmds if "run" in c]
        assert len(run_cmds_502_fb) == 1 and "V6E1" in run_cmds_502_fb[0]
        assert rc_f502_fb != 0
        assert sum_f502_fb["teardown_verified"] is False
        assert sum_f502_fb["status"] == "allocation_timeout_unreconciled"

        # With --no-fallback (single TPU): must STILL run post-run check and keep teardown unverified.
        fast_502_412_cmds.clear()
        rc_f502_nf, sum_f502_nf = run_benchmark(
            BenchConfig(tpus=("v6e1",), out_dir=tmp_path / "fast-502-412-nf"),
            runner=fast_502_412_runner,
        )
        assert rc_f502_nf != 0
        assert sum_f502_nf["teardown_verified"] is False
        assert sum_f502_nf["status"] == "allocation_timeout_unreconciled"

        # Stream-level scanner catches `Response: 502` even in the omitted middle of > 100 KiB stderr
        # before Session READY, while ignoring 502 printed by the user workload AFTER Session READY.
        cap_mid_502 = _BoundedStreamCapture(
            max_capture_bytes=1024, track_session_ready=True, track_uncertain_alloc=True
        )
        cap_mid_502.feed(b"D" * 2048 + b"\ncolab_cli.client - DEBUG - Response: 502 Bad Gateway\n" + b"E" * 2048)
        cap_mid_502.finalize(keep_log=False)
        assert cap_mid_502.saw_pre_ready_uncertain is True

        cap_post_ready_502 = _BoundedStreamCapture(
            max_capture_bytes=1024, track_session_ready=True, track_uncertain_alloc=True
        )
        cap_post_ready_502.feed(
            b"[colab] Session READY (s1). Executing...\n"
            b"user workload hit Response: 502 Bad Gateway\n"
        )
        cap_post_ready_502.finalize(keep_log=False)
        assert cap_post_ready_502.saw_session_ready is True
        assert cap_post_ready_502.saw_pre_ready_uncertain is False

        # When `--help` explicitly omits `--logtostderr` and retries are enabled, 412 is inconclusive.
        nolog_runs: List[str] = []

        def no_logtostderr_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab --auth --config\n", "")
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                nolog_runs.append(cmd[cmd.index("--tpu") + 1])
                return CommandResult(1, "", "[colab] Allocation refused (precondition failed).\n")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        rc_nolog, sum_nolog = run_benchmark(
            BenchConfig(tpus=("v6e1", "v5e1"), out_dir=tmp_path / "no-logtostderr-out"),
            runner=no_logtostderr_runner,
        )
        assert nolog_runs == ["V6E1"], f"Inconclusive 412 without --logtostderr must block fallback: {nolog_runs}"
        assert rc_nolog != 0
        assert sum_nolog["teardown_verified"] is False
        assert sum_nolog["status"] == "allocation_timeout_unreconciled"

        # 64. Preflight / reconciliation `colab sessions` uses a separate throwaway empty config
        #     so an existing explicit `--config` with another account's mappings is NEVER pruned.
        other_acct_dir = tmp_path / "other-acct-store"
        other_acct_dir.mkdir(mode=0o700)
        os.chmod(other_acct_dir, 0o700)
        other_acct_cfg = other_acct_dir / "account-a-sessions.json"
        other_acct_payload = {
            "acct-a-live": {
                "name": "acct-a-live",
                "token": "tok-a",
                "url": "https://example.invalid",
                "endpoint": "m-s-acct-a-ep",
            }
        }
        other_acct_cfg.write_text(json.dumps(other_acct_payload, indent=2) + "\n", encoding="utf-8")
        os.chmod(other_acct_cfg, 0o600)
        sess_configs_seen: List[str] = []

        def pruning_sessions_runner(cmd: Sequence[str]) -> CommandResult:
            if "--help" in cmd:
                return CommandResult(
                    0, "Usage: colab --auth --config --logtostderr --no-public-api\n", ""
                )
            if "version" in cmd:
                return CommandResult(0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 100.00 compute units\nUsage rate: 0.00 / hr\nActive assignments: 0\n",
                    "",
                )
            if "sessions" in cmd:
                cfg_arg = Path(cmd[cmd.index("--config") + 1])
                sess_configs_seen.append(str(cfg_arg))
                # Simulate real Colab CLI 0.8.0 `state.sync_sessions()` pruning all entries not
                # in Account B's empty assignment list!
                cfg_arg.write_text("{}\n", encoding="utf-8")
                return CommandResult(0, "[colab] No active sessions found on server.\n", "")
            if "run" in cmd:
                s_name = cmd[cmd.index("-s") + 1]
                return CommandResult(
                    0,
                    f"{PROBE_JSON_BEGIN}\n{json.dumps(sample_probe)}\n{PROBE_JSON_END}\n",
                    f"[colab] Session READY ({s_name}). Executing...\n",
                )
            if "log" in cmd:
                nb_out = Path(cmd[cmd.index("-o") + 1])
                nb_out.write_text(
                    json.dumps(
                        {
                            "nbformat": 4,
                            "cells": [
                                {"cell_type": "markdown", "source": "- Endpoint: `m-s-acct-b-ep`"},
                                {"cell_type": "code", "source": "print('ok')"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return CommandResult(0, "Exported.\n", "")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        acct_b_out = tmp_path / "acct-b-out"
        rc_ab, sum_ab = run_benchmark(
            BenchConfig(tpus=("v6e1",), config_path=other_acct_cfg, out_dir=acct_b_out),
            runner=pruning_sessions_runner,
        )
        assert rc_ab == 0, sum_ab
        assert len(sess_configs_seen) >= 2
        assert all(Path(c).resolve() != other_acct_cfg.resolve() for c in sess_configs_seen)
        # Account A's session mapping in `other_acct_cfg` survived completely intact!
        assert json.loads(other_acct_cfg.read_text(encoding="utf-8")) == other_acct_payload
        # Throwaway listing config was cleaned up on exit.
        assert not list(acct_b_out.glob("colab-sessions-list.*"))

        # 65. Reject hard-linked `--config` and `<config>.lock` files (st_nlink > 1) pre-flight.
        hl_dir = tmp_path / "hardlink-check"
        hl_dir.mkdir(mode=0o700)
        os.chmod(hl_dir, 0o700)
        hl_backup = hl_dir / "backup-sessions.json"
        hl_backup.write_text("{}", encoding="utf-8")
        os.chmod(hl_backup, 0o600)
        hl_active = hl_dir / "active-sessions.json"
        os.link(hl_backup, hl_active)
        assert os.lstat(hl_active).st_nlink == 2
        rc_hl, sum_hl = run_benchmark(
            BenchConfig(config_path=hl_active, out_dir=tmp_path / "hl-out"),
            runner=forbid_calls_runner,
        )
        assert rc_hl == 2, f"Expected hard-linked config rejection: {sum_hl}"
        assert sum_hl["status"] == "untrusted_config_path"
        assert "hard-linked" in (sum_hl.get("error") or "").lower()

        hl_lock_cfg = hl_dir / "single-link.json"
        hl_lock_cfg.write_text("{}", encoding="utf-8")
        os.chmod(hl_lock_cfg, 0o600)
        hl_lock_orig = hl_dir / "lock-backup"
        hl_lock_orig.write_text("", encoding="utf-8")
        os.chmod(hl_lock_orig, 0o600)
        hl_lock = Path(f"{hl_lock_cfg}.lock")
        os.link(hl_lock_orig, hl_lock)
        rc_hll, sum_hll = run_benchmark(
            BenchConfig(config_path=hl_lock_cfg, out_dir=tmp_path / "hll-out"),
            runner=forbid_calls_runner,
        )
        assert rc_hll == 2, f"Expected hard-linked lock rejection: {sum_hll}"
        assert sum_hll["status"] == "untrusted_config_path"

    print("colab_tpu_bench.py self-test PASSED (65/65 suites).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-room headless Google Colab TPU probe/benchmark harness (wrapping google-colab-cli)."
    )
    parser.add_argument(
        "--colab-bin",
        default="colab",
        help="Path or name of the google-colab-cli executable (default: colab).",
    )
    parser.add_argument(
        "--auth",
        default="adc",
        choices=("adc", "oauth2"),
        help="Authentication mode passed to `colab --auth` (default: adc).",
    )
    parser.add_argument(
        "--tpu",
        default="v6e1",
        choices=tuple(sorted(SUPPORTED_TPUS)),
        help="Primary Colab TPU variant to request (default: v6e1).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back from v6e1 to v5e1 on confirmed capacity/precondition refusals.",
    )
    parser.add_argument(
        "--min-cu-balance",
        type=float,
        default=5.0,
        help="Minimum Compute Units required before provisioning a VM (default: 5.0).",
    )
    parser.add_argument(
        "--max-active-assignments",
        type=int,
        default=None,
        help="Optional cap on active Colab assignments before aborting.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Inactivity timeout (seconds) passed to `colab run` (default: 300.0; 0 disables inactivity timeout).",
    )
    parser.add_argument(
        "--wall-timeout",
        type=float,
        default=None,
        help=(
            "Local wall-clock cap (seconds) for the `colab run` subprocess "
            "(default: --timeout + 180s when --timeout > 0, or 480s when --timeout 0, "
            "to bound stalled kernel startup; pass 0 to explicitly opt out)."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help=(
            "HTTP request timeout (seconds) via COLAB_CLI_REQUEST_TIMEOUT "
            "(explicit flag overrides inherited environment; omitted preserves inherited "
            "COLAB_CLI_REQUEST_TIMEOUT or defaults to 60.0)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Isolated session state JSON path passed to `colab --config`.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Private (0700) directory for summary.json, stdout logs, and exported .ipynb proof.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Explicit session name prefix (a unique run suffix is always appended).",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Optional local Python script to execute instead of the built-in TPU probe.",
    )
    parser.add_argument(
        "--public-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly unsupported until google-colab-cli supports Public API `run`/`usage`/`unassign`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and exit without contacting Colab.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline self-tests using mocked CLI responses.",
    )
    parser.add_argument(
        "script_args",
        nargs="*",
        help="Additional arguments forwarded to the remote script (after `--`).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.dry_run and shutil.which(args.colab_bin) is None:
        print(
            json.dumps(
                {
                    "status": "missing_colab_cli",
                    "error": f"Executable {args.colab_bin!r} not found in PATH. Install via `uv tool install google-colab-cli`.",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    tpus: Tuple[str, ...]
    if args.tpu == "v6e1" and not args.no_fallback:
        tpus = ("v6e1", "v5e1")
    else:
        tpus = (args.tpu,)

    cfg = BenchConfig(
        colab_bin=args.colab_bin,
        auth=args.auth,
        tpus=tpus,
        min_cu_balance=args.min_cu_balance,
        max_active_assignments=args.max_active_assignments,
        timeout_s=args.timeout,
        wall_timeout_s=args.wall_timeout,
        request_timeout_s=args.request_timeout,
        use_no_public_api=not args.public_api,
        config_path=args.config,
        out_dir=args.out_dir,
        script_path=args.script,
        script_args=tuple(args.script_args),
        session_name=args.session,
        dry_run=args.dry_run,
    )
    rc, summary = run_benchmark(cfg)
    print(json.dumps(summary, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
