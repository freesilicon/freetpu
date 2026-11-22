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
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
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

_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\S+)(?:\s+\||$)")


class UntrustedPathError(ValueError):
    """Raised when an output or config path is symlinked or not private to the current user."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


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
    request_timeout_s: float = 60.0
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
    """Write UTF-8 text atomically/safely without following symlinks (mode 0600)."""
    if path.is_symlink():
        raise UntrustedPathError(f"Refusing to write through symlink: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise UntrustedPathError(f"Refusing unsafe target file {path}: {exc}") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise UntrustedPathError(f"Target is not a regular file: {path}")
        if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            raise UntrustedPathError(f"Target file {path} has unexpected owner uid={st.st_uid}")
        fh.write(text)


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
    """Normalize `--config` to an absolute non-symlink path and ensure a safe parent dir."""
    if config_path is None:
        return out_dir / "colab-sessions.json"
    expanded = Path(os.path.expanduser(str(config_path)))
    if expanded.is_symlink():
        raise UntrustedPathError(f"Refusing symlinked --config path: {config_path}")
    normalized = Path(os.path.abspath(str(expanded)))
    _ensure_dir_not_shared_writable(normalized.parent)
    if normalized.is_symlink() or Path(f"{normalized}.lock").is_symlink():
        raise UntrustedPathError(f"Refusing symlinked --config or lock path: {normalized}")
    return normalized


def make_subprocess_runner(request_timeout_s: float = 60.0) -> CommandRunner:
    """Create a non-interactive subprocess runner that bounds `colab` HTTP assign waits."""

    def _run(cmd: Sequence[str]) -> CommandResult:
        env = os.environ.copy()
        env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", str(request_timeout_s))
        env.setdefault("COLAB_CLI_ASSIGN_RETRIES", "1")
        proc = subprocess.run(
            list(cmd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        return CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
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


def is_allocation_timeout_error(returncode: int, stderr: str) -> bool:
    """Return True if allocation timed out with an unknown server-side outcome."""
    if returncode == 0:
        return False
    err_lower = stderr.lower()
    if "[colab] session ready" in err_lower:
        return False
    return "readtimeout" in err_lower or "read timed out" in err_lower


def _contains_oauth_prompt_failure(res: CommandResult) -> bool:
    combined = f"{res.stdout}\n{res.stderr}".lower()
    return any(marker in combined for marker in _OAUTH_PROMPT_MARKERS)


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
    using_default_runner = runner is None
    if runner is None:
        runner = make_subprocess_runner(cfg.request_timeout_s)

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
            out_dir = Path(tempfile.mkdtemp(prefix="freetpu-colab-"))
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
    if sess_before_res.returncode != 0:
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
    maybe_allocated = False
    probe_error: Optional[str] = None
    untrusted_target_error: Optional[str] = None
    artifact_write_error: Optional[str] = None
    export_attempted = False
    export_ok: Optional[bool] = None
    export_returncode: Optional[int] = None
    export_error: Optional[str] = None
    export_stderr_tail: Optional[str] = None

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
        res = runner(run_cmd)
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
        timeout_seen = is_allocation_timeout_error(res.returncode, res.stderr)
        confirmed_pre_alloc = (
            res.returncode != 0
            and not ready_seen
            and not timeout_seen
            and (
                is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)
                or _contains_oauth_prompt_failure(res)
            )
        )
        if res.returncode == 0 or ready_seen or timeout_seen or not confirmed_pre_alloc:
            maybe_allocated = True

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
            final_rc = 0
            chosen_tpu = tpu
            chosen_session = sess
            if generated_script:
                probe_error = validate_builtin_probe(probe_payload)
                if probe_error is not None:
                    final_rc = 1

            nb_path = out_dir / f"{sess}.ipynb"
            if nb_path.is_symlink() or (os.path.lexists(nb_path) and not nb_path.is_file()):
                untrusted_target_error = f"Refusing unsafe notebook target path: {nb_path}"
                break
            export_attempted = True
            log_res = runner([*base_cmd, "log", "-s", sess, "-o", str(nb_path)])
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
            break

        if timeout_seen:
            allocation_timed_out = True
            final_rc = res.returncode
            break

        has_more = idx + 1 < len(cfg.tpus)
        if not (has_more and is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)):
            final_rc = res.returncode
            break

    # 3. Verify server-side TPU assignment release whenever an allocation may have occurred.
    teardown_verified = not maybe_allocated
    leaked_endpoints: List[str] = []
    leaked_sessions: List[str] = []
    ambiguous_endpoints: List[str] = []
    other_new_endpoints: List[str] = []
    teardown_error: Optional[str] = None
    if maybe_allocated:
        sess_after_res = runner([*base_cmd, "sessions"])
        if sess_after_res.returncode != 0:
            teardown_verified = False
            teardown_error = (
                f"Post-run `colab sessions` failed (rc={sess_after_res.returncode}): "
                f"{sess_after_res.stderr.strip()}"
            )
        else:
            names_after, endpoints_after = parse_sessions_output(sess_after_res.stdout)
            new_endpoints = sorted(endpoints_after - endpoints_before)
            if created_endpoint is not None:
                if created_endpoint in endpoints_after:
                    leaked_endpoints.append(created_endpoint)
                other_new_endpoints = [ep for ep in new_endpoints if ep != created_endpoint]
            else:
                ambiguous_endpoints = new_endpoints
            for sname in attempted_sessions:
                if sname in names_after and sname not in names_before:
                    leaked_sessions.append(sname)
            if leaked_endpoints or leaked_sessions:
                teardown_verified = False
                teardown_error = (
                    "Active Colab assignment/session remained after run: "
                    f"endpoints={leaked_endpoints}, sessions={leaked_sessions}"
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
            "teardown_verified": teardown_verified,
            "leaked_endpoints": leaked_endpoints,
            "leaked_sessions": leaked_sessions,
            "ambiguous_endpoints": ambiguous_endpoints,
            "teardown_error": teardown_error,
        }

    if allocation_timed_out and not teardown_verified:
        status = "allocation_timeout_unreconciled"
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
        final_rc = 1
    elif probe_error is not None:
        status = "probe_validation_failed"
        final_rc = 1
    elif export_attempted and not export_ok:
        status = "proof_export_failed"
        final_rc = 1
    elif artifact_write_error is not None:
        status = "artifact_write_failed"
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

        # 4. Leaked TPU endpoint in post-run `colab sessions` fails teardown verification (never returns ok).
        sess_calls = {"n": 0}

        def leaked_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            if "sessions" in cmd_list:
                sess_calls["n"] += 1
                if sess_calls["n"] == 1:
                    return CommandResult(0, "[colab] No active sessions found on server.\n", "")
                return CommandResult(
                    0,
                    "[s] tpu-v5e1-ep1 | Hardware: V5E1 | Shape: STANDARD | Variant: TPU\n",
                    "",
                )
            return fallback_runner(cmd_list)

        rc_leak, sum_leak = run_benchmark(
            BenchConfig(tpus=("v5e1",), out_dir=tmp_path / "leaked"),
            runner=leaked_runner,
        )
        assert rc_leak == 1
        assert sum_leak["status"] == "teardown_verification_failed"
        assert sum_leak["teardown_verified"] is False
        assert "tpu-v5e1-ep1" in sum_leak["leaked_endpoints"]

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

    print("colab_tpu_bench.py self-test PASSED (16/16 suites).")
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
        help="Inactivity timeout (seconds) passed to `colab run` (default: 300.0).",
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
