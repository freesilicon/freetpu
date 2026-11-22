#!/usr/bin/env python3
"""
Clean-room headless Google Colab TPU benchmark and probe harness for FreeTPU.

Wraps the sanctioned `google-colab-cli` (`colab`) control plane to run
reproducible, zero-idle-waste TPU probes (`v6e1` Trillium with graceful
fallback to `v5e1`):

  1. Pre-flight Compute Unit (CU) budget check via `colab usage`.
  2. Per-run session state isolation via `--config <path>` so parallel agents
     never clobber `~/.config/colab-cli/sessions.json`.
  3. Ephemeral provision -> execute -> auto-stop via `colab run` (zero idle CU
     burn even on script error).
  4. Automatic `--tpu v6e1` -> `--tpu v5e1` fallback on HTTP 412 / 400 capacity
     or quota errors.
  5. Replayable notebook proof export via `colab log -s <session> -o <path>.ipynb`.
  6. Offline `--self-test` and `--dry-run` modes so CI passes without Colab
     credentials or live hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROBE_JSON_BEGIN = "=== FREETPU_COLAB_PROBE_JSON_BEGIN ==="
PROBE_JSON_END = "=== FREETPU_COLAB_PROBE_JSON_END ==="

DEFAULT_TPU_ORDER: Tuple[str, ...] = ("v6e1", "v5e1")
SUPPORTED_TPUS = frozenset({"v6e1", "v5e1"})

_FALLBACK_ERROR_MARKERS = (
    "412",
    "400",
    "readtimeout",
    "read timed out",
    "timed out",
    "reason_no_backends",
    "reason_too_many_active_assignments",
    "reason_quota_exceeded_usage_time",
    "reason_runtime_version_unavailable",
    "failed_precondition",
    "allocation refused",
    "backend rejected accelerator",
    "too many active",
    "no backends",
    "capacity",
    "quota or entitlement",
)


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


def make_subprocess_runner(request_timeout_s: float = 60.0) -> CommandRunner:
    """Create a subprocess runner that bounds `colab` HTTP assign waits."""

    def _run(cmd: Sequence[str]) -> CommandResult:
        env = os.environ.copy()
        env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", str(request_timeout_s))
        env.setdefault("COLAB_CLI_ASSIGN_RETRIES", "1")
        proc = subprocess.run(
            list(cmd),
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
    """Parse `colab usage` output into structured CU accounting fields."""
    balance_match = re.search(
        r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)\s+compute units",
        text,
        re.IGNORECASE,
    )
    rate_match = re.search(
        r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*hr",
        text,
        re.IGNORECASE,
    )
    active_match = re.search(
        r"Active assignments:\s*([0-9]+)",
        text,
        re.IGNORECASE,
    )
    return UsageSnapshot(
        balance_cu=float(balance_match.group(1)) if balance_match else None,
        usage_rate_per_hour=float(rate_match.group(1)) if rate_match else None,
        active_assignments=int(active_match.group(1)) if active_match else None,
        raw_output=text.strip(),
    )


def detect_cli_capabilities(
    colab_bin: str, runner: CommandRunner
) -> Dict[str, bool]:
    """Inspect `colab --help` to detect optional flags across CLI releases."""
    res = runner([colab_bin, "--help"])
    combined = f"{res.stdout}\n{res.stderr}"
    return {
        "has_no_public_api": "--no-public-api" in combined,
        "has_config": "--config" in combined,
        "has_auth": "--auth" in combined,
    }


def build_base_cmd(
    cfg: BenchConfig,
    caps: Dict[str, bool],
    isolated_config_path: Path,
) -> List[str]:
    """Build the global `colab` prefix before the subcommand."""
    cmd = [cfg.colab_bin]
    if caps.get("has_auth", True) and cfg.auth:
        cmd.extend(["--auth", cfg.auth])
    if caps.get("has_config", True):
        cmd.extend(["--config", str(isolated_config_path)])
    if cfg.use_no_public_api and caps.get("has_no_public_api", False):
        cmd.append("--no-public-api")
    return cmd


def is_capacity_or_quota_error(
    returncode: int, stdout: str, stderr: str
) -> bool:
    """Return True if a failed `colab run` indicates accelerator unavailability."""
    if returncode == 0:
        return False
    haystack = f"{stderr}\n{stdout}".lower()
    return any(marker in haystack for marker in _FALLBACK_ERROR_MARKERS)


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

        try:
            import jax
            import jax.numpy as jnp

            devices = jax.devices()
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
                "default_backend": jax.default_backend(),
                "device_count": len(devices),
                "devices": dev_info,
            }}

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
                "silicon": devices[0].device_kind if devices else "unknown",
                "dtype": "bf16",
                "m": m,
                "k": k,
                "n": n,
                "hw_us": round(dt_us, 3),
                "gflops": round(gflops, 3),
                "stack": f"jax-{{getattr(jax, '__version__', 'unknown')}}",
            }}
        except Exception as exc:
            report["jax_error"] = str(exc)

        print("{PROBE_JSON_BEGIN}")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("{PROBE_JSON_END}")
        """
    )


def run_benchmark(
    cfg: BenchConfig,
    runner: Optional[CommandRunner] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Execute the pre-flight check, TPU run (with fallback), and log capture."""
    if runner is None:
        runner = make_subprocess_runner(cfg.request_timeout_s)
    out_dir = cfg.out_dir or Path(tempfile.mkdtemp(prefix="freetpu-colab-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    isolated_config = cfg.config_path or (out_dir / "colab-sessions.json")
    session_base = cfg.session_name or f"freetpu-{uuid.uuid4().hex[:8]}"

    script_file = cfg.script_path
    generated_script = False
    if script_file is None:
        script_file = out_dir / "colab_tpu_probe_payload.py"
        script_file.write_text(default_tpu_probe_script(), encoding="utf-8")
        generated_script = True
    elif not script_file.is_file():
        return 2, {"error": f"Script not found: {script_file}"}

    caps = detect_cli_capabilities(cfg.colab_bin, runner)
    base_cmd = build_base_cmd(cfg, caps, isolated_config)

    planned_commands: List[List[str]] = [[*base_cmd, "usage"]]
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
            str(script_file),
            *cfg.script_args,
        ]
        log_cmd = [*base_cmd, "log", "-s", sess, "-o", str(out_dir / f"{sess}.ipynb")]
        planned_commands.append(run_cmd)
        planned_commands.append(log_cmd)

    if cfg.dry_run:
        summary = {
            "status": "dry_run",
            "out_dir": str(out_dir),
            "isolated_config": str(isolated_config),
            "script": str(script_file),
            "generated_script": generated_script,
            "tpus": list(cfg.tpus),
            "planned_commands": planned_commands,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return 0, summary

    # 1. Pre-flight Compute Unit budget check.
    usage_res = runner([*base_cmd, "usage"])
    if usage_res.returncode != 0:
        return 1, {
            "status": "usage_check_failed",
            "returncode": usage_res.returncode,
            "stderr": usage_res.stderr.strip(),
            "stdout": usage_res.stdout.strip(),
        }

    usage = parse_usage_output(usage_res.stdout)
    if usage.balance_cu is not None and usage.balance_cu < cfg.min_cu_balance:
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

    # 2. Ephemeral TPU run with graceful fallback (v6e1 -> v5e1).
    attempts: List[Dict[str, Any]] = []
    final_rc = 1
    chosen_tpu: Optional[str] = None
    chosen_session: Optional[str] = None
    probe_payload: Optional[Dict[str, Any]] = None
    proof_notebook: Optional[Path] = None

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

        if res.returncode == 0:
            final_rc = 0
            chosen_tpu = tpu
            chosen_session = sess
            probe_payload = extract_probe_json(res.stdout)
            (out_dir / f"{sess}.stdout.txt").write_text(res.stdout, encoding="utf-8")

            # Export replayable .ipynb proof transcript via `colab log`.
            nb_path = out_dir / f"{sess}.ipynb"
            log_res = runner([*base_cmd, "log", "-s", sess, "-o", str(nb_path)])
            if log_res.returncode == 0:
                proof_notebook = nb_path
            break

        has_more = idx + 1 < len(cfg.tpus)
        if not (has_more and is_capacity_or_quota_error(res.returncode, res.stdout, res.stderr)):
            final_rc = res.returncode
            (out_dir / f"{sess}.stdout.txt").write_text(res.stdout, encoding="utf-8")
            (out_dir / f"{sess}.stderr.txt").write_text(res.stderr, encoding="utf-8")
            break

    summary = {
        "status": "ok" if final_rc == 0 else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "isolated_config": str(isolated_config),
        "usage_before": asdict(usage),
        "chosen_tpu": chosen_tpu,
        "chosen_session": chosen_session,
        "proof_notebook": str(proof_notebook) if proof_notebook else None,
        "attempts": attempts,
        "probe": probe_payload,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return final_rc, summary


def run_self_test() -> int:
    """Offline self-test verifying CU checks, state isolation, fallback, and proof capture."""
    with tempfile.TemporaryDirectory(prefix="freetpu-colab-selftest-") as tmp:
        tmp_path = Path(tmp)

        # 1. Usage parser check.
        parsed = parse_usage_output(
            "Current balance: 1995.50 compute units\nUsage rate: 2.31/hr\nActive assignments: 1\n"
        )
        assert parsed.balance_cu == 1995.50
        assert parsed.usage_rate_per_hour == 2.31
        assert parsed.active_assignments == 1

        # 2. Low-CU guard aborts before `colab run`.
        seen_cmds: List[List[str]] = []

        def low_cu_runner(cmd: Sequence[str]) -> CommandResult:
            seen_cmds.append(list(cmd))
            if "--help" in cmd:
                return CommandResult(0, "Usage: colab [OPTIONS] --config --auth\n", "")
            if "usage" in cmd:
                return CommandResult(
                    0,
                    "Current balance: 1.50 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n",
                    "",
                )
            raise AssertionError(f"Unexpected command after low-CU abort: {cmd}")

        rc_low, summary_low = run_benchmark(
            BenchConfig(min_cu_balance=5.0, out_dir=tmp_path / "low-cu"),
            runner=low_cu_runner,
        )
        assert rc_low == 1
        assert summary_low["status"] == "cu_budget_insufficient"
        assert not any("run" in c for c in seen_cmds)

        # 3. v6e1 412 fallback -> v5e1 success + log export + isolated --config.
        fallback_cmds: List[List[str]] = []
        iso_cfg = tmp_path / "isolated" / "sessions.json"

        def fallback_runner(cmd: Sequence[str]) -> CommandResult:
            cmd_list = list(cmd)
            fallback_cmds.append(cmd_list)
            if "--help" in cmd_list:
                return CommandResult(0, "Usage: colab --auth --config --no-public-api\n", "")
            if "usage" in cmd_list:
                return CommandResult(
                    0,
                    "Current balance: 500.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n",
                    "",
                )
            if "run" in cmd_list:
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
                            "benchmark": {"silicon": "TPU v5 lite", "hw_us": 123.4},
                        }
                    )
                    return CommandResult(
                        0,
                        f"{PROBE_JSON_BEGIN}\n{payload}\n{PROBE_JSON_END}\n",
                        "[colab] Session stopped.\n",
                    )
            if "log" in cmd_list:
                out_idx = cmd_list.index("-o") + 1
                Path(cmd_list[out_idx]).write_text('{"nbformat": 4, "cells": []}', encoding="utf-8")
                return CommandResult(0, "Exported notebook.\n", "")
            raise AssertionError(f"Unexpected command: {cmd_list}")

        rc_fb, summary_fb = run_benchmark(
            BenchConfig(
                config_path=iso_cfg,
                out_dir=tmp_path / "fallback",
                session_name="tpu-check",
                use_no_public_api=True,
            ),
            runner=fallback_runner,
        )
        assert rc_fb == 0
        assert summary_fb["chosen_tpu"] == "v5e1"
        assert len(summary_fb["attempts"]) == 2
        assert summary_fb["probe"]["benchmark"]["silicon"] == "TPU v5 lite"
        assert summary_fb["proof_notebook"] is not None
        assert Path(summary_fb["proof_notebook"]).is_file()
        for c in fallback_cmds:
            if "--help" not in c:
                assert "--config" in c and str(iso_cfg) in c
                assert "--no-public-api" in c

        # 4. --no-fallback fails immediately when v6e1 is refused.
        rc_nofb, summary_nofb = run_benchmark(
            BenchConfig(
                tpus=("v6e1",),
                out_dir=tmp_path / "nofb",
                session_name="tpu-nofb",
            ),
            runner=fallback_runner,
        )
        assert rc_nofb == 1
        assert summary_nofb["status"] == "failed"
        assert len(summary_nofb["attempts"]) == 1

        # 5. --dry-run produces planned commands without calling usage/run.
        rc_dry, summary_dry = run_benchmark(
            BenchConfig(dry_run=True, out_dir=tmp_path / "dry"),
            runner=low_cu_runner,
        )
        assert rc_dry == 0
        assert summary_dry["status"] == "dry_run"
        assert len(summary_dry["planned_commands"]) == 5

        # 6. ReadTimeout on v6e1 triggers fallback to v5e1.
        assert is_capacity_or_quota_error(
            1, "", "ReadTimeout: HTTPSConnectionPool Read timed out. (read timeout=45.0)"
        )

    print("colab_tpu_bench.py self-test PASSED (5/5 suites).")
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
        help="Do not fall back from v6e1 to v5e1 on capacity/precondition errors.",
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
        help="Directory for summary.json, stdout logs, and exported .ipynb proof.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Explicit session name prefix (auto-generated if omitted).",
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
        help="Use Colab Public API instead of `--no-public-api` tunnel fallback (default: False).",
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
        help="Additional arguments forwarded to the remote script.",
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
