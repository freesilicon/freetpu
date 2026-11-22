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
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PROBE_JSON_BEGIN = "=== FREETPU_COLAB_PROBE_JSON_BEGIN ==="
PROBE_JSON_END = "=== FREETPU_COLAB_PROBE_JSON_END ==="

DEFAULT_TPU_ORDER: Tuple[str, ...] = ("v6e1", "v5e1")
SUPPORTED_TPUS = frozenset({"v6e1", "v5e1"})

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

_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\S+)(?:\s+\||$)")


class UntrustedPathError(ValueError):
    """Raised when an output or config path is symlinked or not private to the current user."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


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
    request_timeout_s: float = 60.0
    teardown_retries: int = 2
    teardown_retry_delay_s: float = 2.0
    use_no_public_api: bool = True
    config_path: Optional[Path] = None
    out_dir: Optional[Path] = None
    script_path: Optional[Path] = None
    script_args: Tuple[str, ...] = ()
    session_name: Optional[str] = None
    dry_run: bool = False


def _is_safe_existing_ancestor(ancestor: Path) -> bool:
    """Verify an already-existing ancestor dir is not a shared writable/unowned trap."""
    if not os.path.lexists(ancestor):
        return True
    st = os.lstat(ancestor)
    if stat.S_ISLNK(st.st_mode):
        # Allow system-owned OS symlinks such as macOS /tmp -> private/tmp or /var -> private/var,
        # but resolve through them before checking ownership/sticky bit.
        if hasattr(os, "geteuid") and st.st_uid != 0 and st.st_uid != os.geteuid():
            return False
        st = os.stat(ancestor)
    if not stat.S_ISDIR(st.st_mode):
        return False
    euid = os.geteuid() if hasattr(os, "geteuid") else st.st_uid
    if st.st_uid != euid and st.st_uid != 0:
        return False
    # If world/group-writable, require sticky bit (e.g., /tmp, /var/tmp) so other users cannot
    # replace or unlink subdirectories created inside it.
    if (st.st_mode & 0o022) != 0 and not (st.st_mode & stat.S_ISVTX):
        return False
    return True


def _validate_existing_ancestor_chain(start_dir: Path) -> None:
    """Validate every existing ancestor directory up to `/` (and symlink targets)."""
    visited: Set[str] = set()
    queue: List[Path] = [start_dir]
    while queue:
        cur = queue.pop()
        while True:
            key = str(cur)
            if key in visited:
                break
            visited.add(key)
            if os.path.lexists(cur):
                if not _is_safe_existing_ancestor(cur):
                    raise UntrustedPathError(
                        f"Refusing path under untrusted ancestor {cur}"
                    )
                if os.path.islink(cur):
                    resolved = Path(os.path.realpath(cur))
                    if str(resolved) not in visited:
                        queue.append(resolved)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent


def _ensure_private_dir(raw_path: Path) -> Path:
    """Ensure `raw_path` exists as a non-symlink directory owned by the user with mode 0700."""
    expanded = Path(os.path.expanduser(str(raw_path)))
    if expanded.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked directory path: {raw_path}")

    abs_path = Path(os.path.abspath(str(expanded)))
    if not os.path.lexists(abs_path):
        # Find deepest existing ancestor and validate the full chain before creating missing 0700 dirs.
        missing: List[Path] = []
        cur = abs_path
        while not os.path.lexists(cur):
            missing.append(cur)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        _validate_existing_ancestor_chain(cur)
        for to_create in reversed(missing):
            try:
                os.mkdir(to_create, 0o700)
            except FileExistsError:
                pass
    _validate_existing_ancestor_chain(abs_path.parent)

    try:
        st = os.lstat(abs_path)
    except OSError as exc:
        raise UntrustedPathError(f"Cannot stat directory {abs_path}: {exc}") from exc

    if stat.S_ISLNK(st.st_mode):
        raise UntrustedPathError(f"Refusing symlinked directory: {abs_path}")
    if not stat.S_ISDIR(st.st_mode):
        raise UntrustedPathError(f"Expected directory, found non-directory: {abs_path}")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UntrustedPathError(
            f"Directory {abs_path} is not owned by current user (uid={st.st_uid})"
        )
    if (st.st_mode & 0o077) != 0:
        raise UntrustedPathError(
            f"Directory {abs_path} must be private (mode 0700); got {oct(stat.S_IMODE(st.st_mode))}"
        )
    return abs_path


def _write_text_nofollow(path: Path, text: str) -> None:
    """Write UTF-8 text via a private temp file and atomically replace `path` (mode 0600)."""
    if path.is_symlink():
        raise UntrustedPathError(f"Refusing to write through symlink: {path}")
    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise UntrustedPathError(f"Refusing unsafe existing target file {path}")
    _ensure_dir_not_shared_writable(path.parent)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
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
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _ensure_dir_not_shared_writable(dir_path: Path) -> Path:
    """Ensure `dir_path` exists as an operator-owned non-symlink dir not writable by others."""
    expanded = Path(os.path.expanduser(str(dir_path)))
    if expanded.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked directory path: {dir_path}")
    abs_dir = Path(os.path.abspath(str(expanded)))
    if not os.path.lexists(abs_dir):
        return _ensure_private_dir(abs_dir)
    _validate_existing_ancestor_chain(abs_dir.parent)
    st = os.lstat(abs_dir)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise UntrustedPathError(f"Refusing non-directory/symlink config parent: {abs_dir}")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UntrustedPathError(
            f"Config parent directory {abs_dir} is not owned by current user (uid={st.st_uid})"
        )
    if (st.st_mode & 0o022) != 0:
        raise UntrustedPathError(
            f"Config parent directory {abs_dir} must not be group/world-writable (mode={oct(stat.S_IMODE(st.st_mode))})"
        )
    return abs_dir


def _normalize_config_path(config_path: Optional[Path], out_dir: Path) -> Path:
    """Normalize explicit or default `--config` to an absolute non-symlink path with safe lock/parent."""
    if config_path is None:
        normalized = Path(os.path.abspath(str(out_dir / "colab-sessions.json")))
    else:
        expanded = Path(os.path.expanduser(str(config_path)))
        if expanded.is_symlink():
            raise UntrustedPathError(f"Refusing symlinked --config path: {config_path}")
        normalized = Path(os.path.abspath(str(expanded)))
    _ensure_dir_not_shared_writable(normalized.parent)
    if normalized.is_symlink() or Path(f"{normalized}.lock").is_symlink():
        raise UntrustedPathError(f"Refusing symlinked --config or lock path: {normalized}")
    return normalized


def _coerce_subprocess_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _ensure_recovery_session_entry(
    config_path: Path,
    session_name: str,
    endpoint: str,
) -> bool:
    """Ensure `config_path` contains a `colab stop`-usable mapping for `session_name -> endpoint`.

    `google-colab-cli` v0.8.0's `colab run` teardown swallows `unassign()` failures and still
    removes the session from local state. Re-seeding a minimal `SessionState`-compatible entry
    with `token_expires_at: null` lets `colab stop -s <session_name>` refresh proxy metadata from
    `list_assignments()` and unassign the stranded endpoint without clobbering other sessions.
    """
    if config_path.is_symlink() or Path(f"{config_path}.lock").is_symlink():
        raise UntrustedPathError(f"Refusing symlinked recovery config/lock path: {config_path}")
    data: Dict[str, Any] = {}
    if os.path.lexists(config_path):
        st = os.lstat(config_path)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise UntrustedPathError(f"Refusing non-regular recovery config path: {config_path}")
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    existing = data.get(session_name)
    if isinstance(existing, dict) and existing.get("endpoint") == endpoint:
        return False
    data[session_name] = {
        "name": session_name,
        "token": "",
        "url": "",
        "endpoint": endpoint,
        "token_expires_at": None,
    }
    _write_text_nofollow(config_path, json.dumps(data, indent=2) + "\n")
    return True


def make_subprocess_runner(
    request_timeout_s: float = 60.0,
    run_wall_timeout_s: Optional[float] = None,
) -> CommandRunner:
    """Create a non-interactive subprocess runner with bounded metadata calls and optional run cap."""

    def _run(cmd: Sequence[str]) -> CommandResult:
        cmd_list = list(cmd)
        env = os.environ.copy()
        env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", str(request_timeout_s))
        env.setdefault("COLAB_CLI_ASSIGN_RETRIES", "1")
        is_run_cmd = "run" in cmd_list
        if is_run_cmd:
            wall_timeout: Optional[float] = (
                run_wall_timeout_s
                if run_wall_timeout_s is not None and run_wall_timeout_s > 0
                else None
            )
        else:
            wall_timeout = max(request_timeout_s * 2.0, 120.0)
        try:
            proc = subprocess.run(
                cmd_list,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=env,
                timeout=wall_timeout,
                check=False,
            )
            return CommandResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            out_txt = _coerce_subprocess_text(exc.stdout)
            err_txt = _coerce_subprocess_text(exc.stderr)
            timeout_label = f"{wall_timeout:.1f}s" if wall_timeout is not None else "configured"
            note = (
                f"[colab_tpu_bench] `colab` subprocess exceeded wall timeout "
                f"({timeout_label}); stopping session and reconciling active sessions."
            )
            combined_err = f"{err_txt}\n{note}".strip() if err_txt else note
            return CommandResult(
                returncode=124,
                stdout=out_txt,
                stderr=combined_err + "\n",
                timed_out=True,
            )

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


def extract_endpoint_from_notebook(nb_path: Path) -> Optional[str]:
    """Extract server endpoint ID recorded in a `colab log -o *.ipynb` session header."""
    if nb_path.is_symlink() or not nb_path.is_file():
        return None
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cells = data.get("cells")
    if not isinstance(cells, list):
        return None
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(str(part) for part in src)
        m = re.search(r"-\s*Endpoint:\s*`?([^`\s\n]+)`?", str(src))
        if m:
            return m.group(1)
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
            "has_public_api": False,
            "has_no_public_api": False,
            "has_config": True,
            "has_auth": True,
        }
    if res.returncode == 127:
        return {
            "cli_found": False,
            "has_public_api": False,
            "has_no_public_api": False,
            "has_config": True,
            "has_auth": True,
        }
    combined = f"{res.stdout}\n{res.stderr}"
    return {
        "cli_found": True,
        "has_public_api": bool(re.search(r"(?<![A-Za-z0-9-])--public-api\b", combined)),
        "has_no_public_api": bool(re.search(r"(?<![A-Za-z0-9-])--no-public-api\b", combined)),
        "has_config": "--config" in combined,
        "has_auth": "--auth" in combined,
    }


def build_base_cmd(
    cfg: BenchConfig,
    caps: Dict[str, bool],
    isolated_config_path: Path,
) -> List[str]:
    """Build the global `colab` prefix before the subcommand."""
    if not cfg.use_no_public_api:
        raise ValueError(
            "`google-colab-cli` still routes `usage`, `sessions`, `run` allocation, and "
            "teardown through the `/tun/m/*` tunnel even when `--public-api` is advertised; "
            "omit `--public-api` here or use Official Colab MCP / Public REST `v1beta` "
            "for Public-API-only environments."
        )
    cmd = [cfg.colab_bin]
    if caps.get("has_auth", True) and cfg.auth:
        cmd.extend(["--auth", cfg.auth])
    if caps.get("has_config", True):
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


def is_allocation_timeout_error(
    returncode: int,
    stderr: str,
    *,
    timed_out: bool = False,
) -> bool:
    """Return True if allocation or setup timed out with an unknown server-side outcome."""
    if returncode == 0 and not timed_out:
        return False
    err_lower = stderr.lower()
    if timed_out:
        return "[colab] session ready" not in err_lower
    if "[colab] session ready" in err_lower:
        return False
    return "readtimeout" in err_lower or "read timed out" in err_lower


def _contains_oauth_prompt_failure(res: CommandResult) -> bool:
    combined = f"{res.stdout}\n{res.stderr}".lower()
    return any(marker in combined for marker in _OAUTH_PROMPT_MARKERS)


def _has_colab_auth_failure(res: CommandResult) -> bool:
    """Return True if `colab` emitted an authentication/credential failure (even on exit 0)."""
    combined = f"{res.stdout}\n{res.stderr}".lower()
    return any(marker in combined for marker in _COLAB_AUTH_ERROR_MARKERS)


def extract_probe_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract JSON payload emitted between FreeTPU probe markers."""
    pattern = re.compile(
        re.escape(PROBE_JSON_BEGIN) + r"\s*(.*?)\s*" + re.escape(PROBE_JSON_END),
        re.DOTALL,
    )
    match = pattern.search(stdout)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return None


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
    if cfg.wall_timeout_s is not None and cfg.wall_timeout_s < 0:
        return 2, {
            "status": "invalid_timeout",
            "error": f"`--wall-timeout` must be >= 0 (got {cfg.wall_timeout_s!r}).",
        }

    using_default_runner = runner is None
    if runner is None:
        effective_wall_timeout = (
            cfg.wall_timeout_s
            if cfg.wall_timeout_s is not None and cfg.wall_timeout_s > 0
            else None
        )
        runner = make_subprocess_runner(
            cfg.request_timeout_s,
            run_wall_timeout_s=effective_wall_timeout,
        )

    for arg in cfg.script_args:
        if arg == "--env" or arg.startswith("--env="):
            return 2, {
                "status": "unsupported_script_arg",
                "error": (
                    "Forwarding `--env` via script arguments is blocked because `colab run` "
                    "parses `--env` even after `--`."
                ),
            }

    if cfg.session_name is not None and not _SAFE_SESSION_RE.match(cfg.session_name):
        return 2, {
            "status": "invalid_session_name",
            "error": (
                f"Session prefix {cfg.session_name!r} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}."
            ),
        }

    try:
        if cfg.out_dir is None:
            env_tmp = os.environ.get("TMPDIR")
            tmp_root = Path(os.path.abspath(env_tmp if env_tmp else tempfile.gettempdir()))
            _validate_existing_ancestor_chain(tmp_root)
            created_tmp = Path(tempfile.mkdtemp(prefix="freetpu-colab-", dir=str(tmp_root)))
            try:
                out_dir = _ensure_private_dir(created_tmp)
            except Exception:
                shutil.rmtree(created_tmp, ignore_errors=True)
                raise
        else:
            out_dir = _ensure_private_dir(cfg.out_dir)
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_out_dir", "error": str(exc)}

    try:
        isolated_config = _normalize_config_path(cfg.config_path, out_dir)
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_config_path", "error": str(exc)}

    session_prefix = cfg.session_name or "freetpu"
    session_base = f"{session_prefix}-{uuid.uuid4().hex[:8]}"

    script_file = cfg.script_path
    generated_script = False
    if script_file is None:
        script_file = out_dir / "colab_tpu_probe_payload.py"
        try:
            _write_text_nofollow(script_file, default_tpu_probe_script())
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        generated_script = True
    elif script_file.is_symlink() or not script_file.is_file():
        return 2, {"status": "invalid_script", "error": f"Invalid script path: {script_file}"}

    caps = detect_cli_capabilities(cfg.colab_bin, runner)
    if not cfg.dry_run and not caps.get("cli_found", True):
        return 2, {
            "status": "missing_colab_cli",
            "error": (
                f"Executable {cfg.colab_bin!r} not found in PATH. Install via `uv tool install google-colab-cli`."
            ),
        }

    try:
        base_cmd = build_base_cmd(cfg, caps, isolated_config)
    except ValueError as exc:
        return 2, {"status": "unsupported_public_api_mode", "error": str(exc)}

    planned_commands: List[List[str]] = [
        [*base_cmd, "usage"],
        [*base_cmd, "sessions"],
    ]
    for idx, tpu in enumerate(cfg.tpus):
        sess = f"{session_base}-{tpu}" if idx > 0 else session_base
        run_cmd = [
            *base_cmd,
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
    planned_commands.append([*base_cmd, "sessions"])

    if cfg.dry_run:
        summary = {
            "status": "dry_run",
            "colab_cli_found": bool(caps.get("cli_found", True)),
            "out_dir": str(out_dir),
            "isolated_config": str(isolated_config),
            "script": str(script_file),
            "generated_script": generated_script,
            "session_base": session_base,
            "tpus": list(cfg.tpus),
            "timeout_s": cfg.timeout_s,
            "wall_timeout_s": cfg.wall_timeout_s,
            "planned_commands": planned_commands,
        }
        try:
            _write_text_nofollow(
                out_dir / "summary.json",
                json.dumps(summary, indent=2) + "\n",
            )
        except UntrustedPathError as exc:
            return 2, {"status": "untrusted_out_dir", "error": str(exc)}
        return 0, summary

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

    sess_before_res = runner([*base_cmd, "sessions"])
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
    allocation_timed_out = False
    wall_timed_out = False
    interrupted = False
    maybe_allocated = False
    probe_error: Optional[str] = None
    untrusted_target_error: Optional[str] = None
    artifact_write_error: Optional[str] = None
    export_attempted = False
    export_ok: Optional[bool] = None
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

    def _run_stop_cmd(sess_name: str) -> Tuple[CommandResult, Optional[str]]:
        stop_res = runner([*base_cmd, "stop", "-s", sess_name])
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

    def _export_session_log(sess_name: str) -> None:
        nonlocal export_attempted, export_ok, export_returncode, export_error, export_stderr_tail
        nonlocal proof_notebook, created_endpoint, untrusted_target_error
        nb_path = out_dir / f"{sess_name}.ipynb"
        if nb_path.is_symlink() or (os.path.lexists(nb_path) and not nb_path.is_file()):
            untrusted_target_error = f"Refusing unsafe notebook target path: {nb_path}"
            return
        export_attempted = True
        log_res = runner([*base_cmd, "log", "-s", sess_name, "-o", str(nb_path)])
        export_returncode = log_res.returncode
        if log_res.stderr.strip():
            export_stderr_tail = log_res.stderr.strip()[-1000:]
        if log_res.returncode == 0 and not nb_path.is_symlink() and nb_path.is_file():
            proof_notebook = nb_path
            created_endpoint = extract_endpoint_from_notebook(nb_path)
            export_ok = True
        else:
            export_ok = False
            if log_res.returncode != 0:
                diag = log_res.stderr.strip() or log_res.stdout.strip() or "no diagnostic output"
                export_error = f"`colab log` failed (rc={log_res.returncode}): {diag}"
            else:
                export_error = f"`colab log` exited 0 without creating notebook at {nb_path}"

    for idx, tpu in enumerate(cfg.tpus):
        sess = f"{session_base}-{tpu}" if idx > 0 else session_base
        attempted_sessions.append(sess)
        run_cmd = [
            *base_cmd,
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
        try:
            res = runner(run_cmd)
        except KeyboardInterrupt:
            interrupted = True
            maybe_allocated = True
            final_rc = 130
            if chosen_tpu is None:
                chosen_tpu = tpu
                chosen_session = sess
            attempts.append(
                {
                    "tpu": tpu,
                    "session": sess,
                    "returncode": 130,
                    "stderr_tail": "[colab_tpu_bench] Interrupted by operator (KeyboardInterrupt).",
                }
            )
            # CPython's subprocess.run kills `colab run` on Ctrl+C before its teardown can finish;
            # run bounded cleanup using the attempted session name and isolated config.
            forced_stop["attempted"] = True
            forced_stop["session"] = sess
            stop_res, stop_err = _run_stop_cmd(sess)
            forced_stop["returncode"] = stop_res.returncode
            if stop_res.stderr.strip():
                forced_stop["stderr_tail"] = stop_res.stderr.strip()[-1000:]
            forced_stop["error"] = stop_err
            _export_session_log(sess)
            break

        attempt_rec = {
            "tpu": tpu,
            "session": sess,
            "returncode": res.returncode,
            "stderr_tail": res.stderr.strip()[-1000:],
        }
        attempts.append(attempt_rec)

        parsed_probe = extract_probe_json(res.stdout)
        if parsed_probe is not None:
            probe_payload = parsed_probe

        ready_seen = "[colab] session ready" in res.stderr.lower()
        wall_timeout_seen = is_wall_timeout_error(res)
        timeout_seen = is_allocation_timeout_error(
            res.returncode, res.stderr, timed_out=res.timed_out
        )
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

        # If `subprocess.run` killed `colab run` on wall-clock timeout, its own
        # `finally: _teardown(...)` handler never ran; explicitly stop the session
        # using the exact same isolated `--config` before exporting logs or checking sessions.
        if wall_timeout_seen:
            wall_timed_out = True
            forced_stop["attempted"] = True
            forced_stop["session"] = sess
            stop_res, stop_err = _run_stop_cmd(sess)
            forced_stop["returncode"] = stop_res.returncode
            if stop_res.stderr.strip():
                forced_stop["stderr_tail"] = stop_res.stderr.strip()[-1000:]
            forced_stop["error"] = stop_err

        try:
            if res.stdout:
                _write_text_nofollow(out_dir / f"{sess}.stdout.txt", res.stdout)
            if res.returncode != 0 and res.stderr:
                _write_text_nofollow(out_dir / f"{sess}.stderr.txt", res.stderr)
        except UntrustedPathError as exc:
            untrusted_target_error = str(exc)
            break
        except OSError as exc:
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

        has_more = idx + 1 < len(cfg.tpus)
        if not (has_more and is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)):
            break

    # 3. Verify server-side TPU assignment release whenever an allocation may have occurred,
    #    re-seeding local recovery state and retrying release if this run's assignment is still active.
    teardown_verified = not maybe_allocated
    leaked_endpoints: List[str] = []
    leaked_sessions: List[str] = []
    ambiguous_endpoints: List[str] = []
    other_new_endpoints: List[str] = []
    teardown_error: Optional[str] = None

    def _inspect_active_sessions() -> Tuple[
        Optional[str], List[str], List[str], List[str], List[str]
    ]:
        sess_res = runner([*base_cmd, "sessions"])
        if sess_res.returncode != 0 or _has_colab_auth_failure(sess_res):
            diag = sess_res.stderr.strip() or sess_res.stdout.strip()
            if sess_res.returncode != 0:
                return (
                    f"Post-run `colab sessions` failed (rc={sess_res.returncode}): {diag}",
                    [],
                    [],
                    [],
                    [],
                )
            return (
                f"Post-run `colab sessions` exited 0 with authentication error: {diag}",
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
        return None, cur_leaked_eps, cur_leaked_sess, cur_ambig_eps, cur_other_eps

    if maybe_allocated:
        (
            query_err,
            leaked_endpoints,
            leaked_sessions,
            ambiguous_endpoints,
            other_new_endpoints,
        ) = _inspect_active_sessions()
        if query_err is not None:
            teardown_verified = False
            teardown_error = query_err
        elif leaked_endpoints or leaked_sessions:
            recovery_session = (
                chosen_session
                or (leaked_sessions[0] if leaked_sessions else None)
                or (attempted_sessions[-1] if attempted_sessions else session_base)
            )
            max_retries = max(int(cfg.teardown_retries), 0)
            for retry_idx in range(1, max_retries + 1):
                seeded = False
                seed_err: Optional[str] = None
                if created_endpoint is not None and created_endpoint in leaked_endpoints:
                    try:
                        seeded = _ensure_recovery_session_entry(
                            isolated_config, recovery_session, created_endpoint
                        )
                    except (UntrustedPathError, OSError) as exc:
                        seed_err = str(exc)

                stop_targets = list(
                    dict.fromkeys(
                        [
                            *leaked_sessions,
                            *([recovery_session] if leaked_endpoints else []),
                        ]
                    )
                )
                for target_sess in stop_targets:
                    retry_res, retry_err = _run_stop_cmd(target_sess)
                    release_retries.append(
                        {
                            "attempt": retry_idx,
                            "session": target_sess,
                            "endpoint": created_endpoint,
                            "reseeded_config": seeded,
                            "seed_error": seed_err,
                            "returncode": retry_res.returncode,
                            "error": retry_err,
                            "stderr_tail": retry_res.stderr.strip()[-1000:]
                            if retry_res.stderr.strip()
                            else None,
                        }
                    )

                (
                    query_err,
                    leaked_endpoints,
                    leaked_sessions,
                    ambiguous_endpoints,
                    other_new_endpoints,
                ) = _inspect_active_sessions()
                if query_err is not None:
                    teardown_verified = False
                    teardown_error = query_err
                    break
                if not leaked_endpoints and not leaked_sessions:
                    if ambiguous_endpoints:
                        teardown_verified = False
                        teardown_error = (
                            "Run endpoint is unknown and new Colab assignment(s) appeared after run: "
                            f"ambiguous_endpoints={ambiguous_endpoints}"
                        )
                    else:
                        teardown_verified = True
                        teardown_error = None
                    break
                if retry_idx < max_retries and cfg.teardown_retry_delay_s > 0:
                    try:
                        time.sleep(cfg.teardown_retry_delay_s)
                    except KeyboardInterrupt:
                        interrupted = True
                        final_rc = 130
                        break

            if query_err is None and (leaked_endpoints or leaked_sessions):
                if created_endpoint is not None and created_endpoint in leaked_endpoints:
                    try:
                        _ensure_recovery_session_entry(
                            isolated_config, recovery_session, created_endpoint
                        )
                    except (UntrustedPathError, OSError):
                        pass
                recovery_command = shlex.join([*base_cmd, "stop", "-s", recovery_session])
                teardown_verified = False
                teardown_error = (
                    "Active Colab assignment/session remained after release retries: "
                    f"endpoints={leaked_endpoints}, sessions={leaked_sessions}; "
                    f"recovery config preserved at {isolated_config} "
                    f"(run: `{recovery_command}`)"
                )
        elif ambiguous_endpoints:
            teardown_verified = False
            teardown_error = (
                "Run endpoint is unknown and new Colab assignment(s) appeared after run: "
                f"ambiguous_endpoints={ambiguous_endpoints}"
            )
        else:
            teardown_verified = True

    if untrusted_target_error is not None:
        return 2, {
            "status": "untrusted_out_dir",
            "error": untrusted_target_error,
            "forced_stop": forced_stop,
            "release_retries": release_retries,
            "recovery_command": recovery_command,
            "teardown_verified": teardown_verified,
            "leaked_endpoints": leaked_endpoints,
            "leaked_sessions": leaked_sessions,
            "ambiguous_endpoints": ambiguous_endpoints,
            "teardown_error": teardown_error,
        }

    if interrupted:
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

    summary = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "isolated_config": str(isolated_config),
        "usage_before": asdict(usage),
        "chosen_tpu": chosen_tpu,
        "chosen_session": chosen_session,
        "created_endpoint": created_endpoint,
        "proof_notebook": str(proof_notebook) if proof_notebook else None,
        "proof_export": {
            "attempted": export_attempted,
            "ok": export_ok,
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
    try:
        _write_text_nofollow(
            out_dir / "summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
    except UntrustedPathError as exc:
        return 2, {"status": "untrusted_out_dir", "error": str(exc)}
    except OSError as exc:
        summary["artifact_write_error"] = summary["artifact_write_error"] or str(exc)
        if summary["status"] == "ok":
            summary["status"] = "artifact_write_failed"
            final_rc = 1
    return final_rc, summary


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
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
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
                            }
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
                if "--help" not in c:
                    assert "--config" in c and sum_fb["isolated_config"] in c
                    assert "--no-public-api" in c

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
        preserved_cfg = json.loads(Path(sum_leak["isolated_config"]).read_text(encoding="utf-8"))
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
        assert sum_to["status"] == "allocation_timeout"

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

        # 18. Default derived config / lock symlinks inside reused out_dir are rejected.
        reused_dir = tmp_path / "reused-default-cfg"
        reused_dir.mkdir(mode=0o700)
        os.chmod(reused_dir, 0o700)
        (reused_dir / "colab-sessions.json.lock").symlink_to(tmp_path / "outside.lock")
        rc_dlock, sum_dlock = run_benchmark(
            BenchConfig(dry_run=True, out_dir=reused_dir, config_path=None),
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
            if "--help" in cmd_list or "usage" in cmd_list:
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

        # 23. Pre-READY wall-clock timeout (rc=124) also attempts `colab stop -s <sess>` before
        #     verifying release, and reports `allocation_timeout` without falling back.
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
        assert sum_pwt["status"] == "allocation_timeout"
        assert sum_pwt["teardown_verified"] is True
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
            if "--help" in cmd_list or "usage" in cmd_list:
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
        unrec_cfg = json.loads(Path(sum_intu["isolated_config"]).read_text(encoding="utf-8"))
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
        assert sum_tr["release_retries"][0]["reseeded_config"] is True
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

    print("colab_tpu_bench.py self-test PASSED (29/29 suites).")
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
            "Optional local wall-clock cap (seconds) for the `colab run` subprocess "
            "(default: unset / 0 = no local wall-clock cap; independent of --timeout)."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout (seconds) via COLAB_CLI_REQUEST_TIMEOUT (default: 60.0).",
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
