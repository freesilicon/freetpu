"""Clean-room headless Google Colab TPU launcher for FreeTPU Tile DSL kernels.

Wraps `google-colab-cli` (`colab`) with:
  1. Isolated session state (`--config <isolated-path>`).
  2. CU pre-flight budget check (`colab --auth=adc usage`).
  3. Single-file self-extracting bundle of `freetpu_tile/` + the target example script
     so the exact unmodified `@tile_kernel` source runs on Colab TPU and auto-stops cleanly.
  4. Automatic `v6e1` -> `v5e1` fallback on capacity / quota errors (only before Session READY).
  5. Replayable `.ipynb` transcript capture via `colab log -s <session> -o <proof>.ipynb`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_ALLOCATION_ERROR_MARKERS = (
    "allocation refused",
    "backend rejected accelerator",
    "toomanyassignmentserror",
    "failed_precondition",
    "precondition failed",
    "reason_no_backends",
    "reason_too_many_active_assignments",
    "reason_quota_exceeded_usage_time",
    "reason_runtime_version_unavailable",
    "quota or entitlement",
    "too many active sessions",
    "no backends",
    "service unavailable",
)


@dataclass(frozen=True)
class CmdRes:
    returncode: int
    stdout: str
    stderr: str


RunnerFn = Callable[[Sequence[str]], CmdRes]


def default_colab_bin() -> str:
    local_bin = Path.home() / ".local" / "bin" / "colab"
    if local_bin.is_file():
        return str(local_bin)
    found = shutil.which("colab")
    return found or "colab"


def _make_runner(request_timeout_s: float = 60.0) -> RunnerFn:
    def _run(cmd: Sequence[str]) -> CmdRes:
        env = os.environ.copy()
        env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", str(request_timeout_s))
        env.setdefault("COLAB_CLI_ASSIGN_RETRIES", "1")
        proc = subprocess.run(list(cmd), capture_output=True, text=True, env=env, check=False)
        return CmdRes(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

    return _run


def parse_usage_cu(text: str) -> Optional[float]:
    m = re.search(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)\s+compute units", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def is_capacity_error(returncode: int, stderr: str) -> bool:
    if returncode == 0:
        return False
    low = stderr.lower()
    if "[colab] session ready" in low:
        return False
    return any(m in low for m in _ALLOCATION_ERROR_MARKERS)


def build_remote_bundle(
    script_path: Path,
    pkg_root: Optional[Path] = None,
    script_args: Sequence[str] = (),
) -> str:
    """Create a clean-room single-file wrapper bundling `freetpu_tile/` + `script_path` + `script_args`."""
    pkg_dir = pkg_root or Path(__file__).resolve().parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for py_file in sorted(pkg_dir.rglob("*.py")):
            rel = py_file.relative_to(pkg_dir.parent)
            if "__pycache__" in rel.parts:
                continue
            tf.add(py_file, arcname=str(rel))
    b64_pkg = base64.b64encode(buf.getvalue()).decode("ascii")
    script_src = script_path.read_text(encoding="utf-8")
    b64_script = base64.b64encode(script_src.encode("utf-8")).decode("ascii")
    embedded_args = [str(a) for a in script_args]

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import base64
        import io
        import os
        import sys
        import tarfile
        import tempfile
        from pathlib import Path

        PKG_B64 = {b64_pkg!r}
        SCRIPT_B64 = {b64_script!r}
        SCRIPT_ARGS = {embedded_args!r}

        with tempfile.TemporaryDirectory(prefix="freetpu-colab-") as tmp:
            tmp_p = Path(tmp)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64)), mode="r:gz") as tf:
                tf.extractall(tmp_p)
            script_file = tmp_p / {script_path.name!r}
            script_file.write_bytes(base64.b64decode(SCRIPT_B64))
            sys.path.insert(0, str(tmp_p))
            sys.argv = [str(script_file), *SCRIPT_ARGS, *sys.argv[1:]]
            ns = {{"__name__": "__main__", "__file__": str(script_file)}}
            exec(compile(script_file.read_text(encoding="utf-8"), str(script_file), "exec"), ns, ns)
        """
    )


def run_colab_tile_kernel(
    script_path: Path,
    *,
    script_args: Sequence[str] = ("--target", "cloud_tpu_pallas"),
    tpus: Tuple[str, ...] = ("v6e1", "v5e1"),
    colab_bin: Optional[str] = None,
    auth: str = "adc",
    config_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    session_name: Optional[str] = None,
    min_cu_balance: float = 5.0,
    timeout_s: float = 300.0,
    dry_run: bool = False,
    runner: Optional[RunnerFn] = None,
) -> Tuple[int, Dict[str, Any]]:
    if not script_path.is_file():
        return 2, {"status": "missing_script", "error": f"Script not found: {script_path}"}

    resolved_bin = colab_bin or default_colab_bin()
    exec_runner = runner or _make_runner()
    work_dir = out_dir or Path(tempfile.mkdtemp(prefix="freetpu-colab-run-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    iso_cfg = config_path or (work_dir / "colab-sessions.json")
    sess_base = session_name or f"freetpu-tile-{uuid.uuid4().hex[:8]}"

    bundle_path = work_dir / f"bundle_{script_path.stem}.py"
    bundle_path.write_text(
        build_remote_bundle(script_path, script_args=script_args),
        encoding="utf-8",
    )

    prefix = [resolved_bin, "--auth", auth, "--config", str(iso_cfg)]
    planned: List[List[str]] = [[*prefix, "usage"]]
    for idx, tpu in enumerate(tpus):
        sess = f"{sess_base}-{tpu}" if idx > 0 else sess_base
        planned.append([
            *prefix,
            "run",
            "-s",
            sess,
            "--tpu",
            tpu.upper(),
            "--timeout",
            str(timeout_s),
            str(bundle_path),
        ])
        planned.append([*prefix, "log", "-s", sess, "-o", str(work_dir / f"{sess}.ipynb")])

    if dry_run:
        summary = {
            "status": "dry_run",
            "script": str(script_path),
            "script_args": list(script_args),
            "bundle_path": str(bundle_path),
            "isolated_config": str(iso_cfg),
            "tpus": list(tpus),
            "planned_commands": planned,
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0, summary

    usage_res = exec_runner([*prefix, "usage"])
    if usage_res.returncode != 0:
        return 1, {"status": "usage_failed", "stderr": usage_res.stderr.strip(), "stdout": usage_res.stdout.strip()}

    cu_bal = parse_usage_cu(usage_res.stdout)
    if cu_bal is not None and cu_bal < min_cu_balance:
        return 1, {"status": "cu_budget_insufficient", "balance_cu": cu_bal, "min_cu_balance": min_cu_balance}

    attempts: List[Dict[str, Any]] = []
    chosen_tpu: Optional[str] = None
    chosen_session: Optional[str] = None
    notebook_path: Optional[Path] = None
    stdout_path: Optional[Path] = None
    final_rc = 1

    for idx, tpu in enumerate(tpus):
        sess = f"{sess_base}-{tpu}" if idx > 0 else sess_base
        cmd = [
            *prefix,
            "run",
            "-s",
            sess,
            "--tpu",
            tpu.upper(),
            "--timeout",
            str(timeout_s),
            str(bundle_path),
        ]
        res = exec_runner(cmd)
        attempts.append({
            "tpu": tpu,
            "session": sess,
            "returncode": res.returncode,
            "stderr_tail": res.stderr.strip()[-1000:],
        })
        if res.returncode == 0:
            final_rc = 0
            chosen_tpu = tpu
            chosen_session = sess
            stdout_path = work_dir / f"{sess}.stdout.txt"
            stdout_path.write_text(res.stdout, encoding="utf-8")
            nb_out = work_dir / f"{sess}.ipynb"
            log_res = exec_runner([*prefix, "log", "-s", sess, "-o", str(nb_out)])
            if log_res.returncode == 0 and nb_out.is_file():
                notebook_path = nb_out
            break
        if idx + 1 >= len(tpus) or not is_capacity_error(res.returncode, res.stderr):
            final_rc = res.returncode
            (work_dir / f"{sess}.stdout.txt").write_text(res.stdout, encoding="utf-8")
            (work_dir / f"{sess}.stderr.txt").write_text(res.stderr, encoding="utf-8")
            break

    summary = {
        "status": "ok" if final_rc == 0 else "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "balance_cu_before": cu_bal,
        "isolated_config": str(iso_cfg),
        "chosen_tpu": chosen_tpu,
        "chosen_session": chosen_session,
        "proof_notebook": str(notebook_path) if notebook_path else None,
        "stdout_log": str(stdout_path) if stdout_path else None,
        "attempts": attempts,
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return final_rc, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    forwarded_after_sep: Tuple[str, ...] = ()
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        forwarded_after_sep = tuple(raw_argv[sep + 1 :])
        raw_argv = raw_argv[:sep]

    ap = argparse.ArgumentParser(description="Run a FreeTPU @tile_kernel script headlessly on Colab TPU.")
    ap.add_argument("script", type=Path, help="Local Python @tile_kernel script to run on Colab TPU.")
    ap.add_argument("--tpu", default="v6e1", choices=("v6e1", "v5e1"))
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--config", type=Path, default=None, help="Isolated colab-cli session state JSON path.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory for .ipynb proof and logs.")
    ap.add_argument("--session", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("script_args", nargs="*", help="Arguments passed to the remote script (prefer passing after `--`).")
    args = ap.parse_args(raw_argv)

    tpus = (args.tpu,) if (args.no_fallback or args.tpu != "v6e1") else ("v6e1", "v5e1")
    combined_args = (*args.script_args, *forwarded_after_sep)
    extra_args = combined_args if combined_args else ("--target", "cloud_tpu_pallas")
    rc, summary = run_colab_tile_kernel(
        args.script,
        script_args=extra_args,
        tpus=tpus,
        config_path=args.config,
        out_dir=args.out_dir,
        session_name=args.session,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
