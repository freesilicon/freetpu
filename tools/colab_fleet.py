#!/usr/bin/env python3
"""Clean-room 7-Slot Google Colab Fleet Dispatcher, Dual Sync & Two-Stage CU Governor.

Extends `tools/colab_tpu_bench.py` safety patterns to orchestrate up to 7 concurrent
ephemeral Colab lanes (GPU + TPU + High-RAM CPU) for public OSS research, training,
and verification workloads with zero idle burn and zero private-data leakage:

  - Slot Caps & Pre-Existing Session Isolation:
      Respects all pre-existing sessions on the account, caps total active assignments
      at `--max-total-slots` (default 7, empirical AI Ultra ceiling), GPU slots at
      `--max-gpu-slots` (default 3, ordered fallback H100 -> A100_HM -> A100 -> L4 -> T4_HM),
      TPU slots at `--max-tpu-slots` (default 2, V6E1 -> V5E1), and fills remaining slots
      with High-RAM CPU (`0.26 CU/hr`, 8 vCPU / 51 GiB) verifier/coordinator workers.
      Never stops, execs on, or touches any session not prefixed `fleet-`.
  - Zero Retry-Hammering:
      Sets `COLAB_CLI_ASSIGN_RETRIES=0` and applies cooldown backoff on HTTP 503/412/429.
  - Dual Pluggable Sync Backend:
      * `--sync local` (default): zero cloud keys on Colab; syncs via `colab upload` /
        `colab download` to a private `0700` local staging directory.
      * `--sync s3`: generic S3-compatible object store (Cloudflare R2, GCS HMAC, MinIO,
        AWS S3) via stdlib AWS SigV4 using env vars (`FREETPU_S3_ENDPOINT`,
        `FREETPU_S3_BUCKET`, `FREETPU_S3_ACCESS_KEY_ID`, `FREETPU_S3_SECRET_ACCESS_KEY`);
        secrets are never written to logs, `STATE.md`, `events.jsonl`, or `PROOF.md`.
  - Two-Stage CU Governor:
      * Sprint mode (`balance_cu > floor_cu`, default `500.0` CUs): runs up to 7 slots
        concurrently to burn banked rollover CUs productively down to the floor.
      * Steady-state mode (`balance_cu <= floor_cu`): auto-throttles to
        `--steady-daily-cap-cu` (default `65.0` CUs/day ≈ 2,000 CU/mo rollover) and
        idles at 0 CUs when no work is admitted.
  - Durable Telemetry & Guaranteed Teardown:
      Writes `STATE.md`, `events.jsonl`, `heartbeat`, and `PROOF.md`, and guarantees
      `fleet-*` session cleanup on normal exit, SIGINT, or SIGTERM.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_GPU_FALLBACK: Tuple[str, ...] = ("H100", "A100_HM", "A100", "L4", "T4_HM", "T4")
DEFAULT_TPU_FALLBACK: Tuple[str, ...] = ("V6E1", "V5E1")
DEFAULT_CPU_FALLBACK: Tuple[str, ...] = ("CPU_HM", "CPU")

SKU_CLI_FLAGS: Dict[str, Tuple[str, ...]] = {
    "H100": ("--gpu", "H100"),
    "A100_HM": ("--gpu", "A100", "--high-mem"),
    "A100": ("--gpu", "A100"),
    "G4": ("--gpu", "G4"),
    "L4": ("--gpu", "L4"),
    "T4_HM": ("--gpu", "T4", "--high-mem"),
    "T4": ("--gpu", "T4"),
    "V6E1": ("--tpu", "V6E1"),
    "V5E1": ("--tpu", "V5E1"),
    "CPU_HM": ("--high-mem",),
    "CPU": (),
}

SKU_KIND: Dict[str, str] = {
    "H100": "GPU",
    "A100_HM": "GPU",
    "A100": "GPU",
    "G4": "GPU",
    "L4": "GPU",
    "T4_HM": "GPU",
    "T4": "GPU",
    "V6E1": "TPU",
    "V5E1": "TPU",
    "CPU_HM": "CPU",
    "CPU": "CPU",
}

SKU_EST_CU_HR: Dict[str, float] = {
    "H100": 13.0,
    "A100_HM": 11.5,
    "A100": 8.16,
    "G4": 9.0,
    "L4": 2.33,
    "T4_HM": 1.87,
    "T4": 1.24,
    "V6E1": 4.5,
    "V5E1": 3.26,
    "CPU_HM": 0.26,
    "CPU": 0.08,
}

SESSION_LINE_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s+\|\s+Hardware:\s+(?P<hw>[^|]+)\|\s+Shape:\s+(?P<shape>[^|]+)\|\s+Variant:\s+(?P<variant>\S+)"
)
ZONE_TOKEN_RE = re.compile(r"-(?P<zone>(?:us|eu|as|na|sa|au|me)[a-z0-9]{2,8})-")

FORBIDDEN_PRIVATE_MARKERS = (
    "archive.sqlite",
    "credentials.json",
    "secrets.json",
    ".env",
    "id_rsa",
    "id_ed25519",
    "1password",
    "cookies.sqlite",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_day_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def redact_secrets(text: str) -> str:
    """Redact bearer tokens, cookies, AWS/S3 keys, and auth headers from diagnostic text."""
    s = re.sub(r"(?i)(bearer|token|cookie|authorization|x-goog-colab-token|secret|signature)\s*[:=]\s*\S+", r"\1=REDACTED", text)
    s = re.sub(r"AKIA[0-9A-Z]{16}", "AKIA_REDACTED", s)
    return s


def assert_public_path(path_str: str) -> None:
    low = path_str.lower()
    extra = [x.strip().lower() for x in os.environ.get("FREETPU_EXTRA_DENY_PATTERNS", "").split(",") if x.strip()]
    for marker in (*FORBIDDEN_PRIVATE_MARKERS, *extra):
        if marker in low:
            raise ValueError(f"Public-only guardrail blocked path containing {marker!r}: {path_str}")


# ---------------------------------------------------------------------------
# Two-Stage CU Governor (Sprint -> Floor 500 CU -> Steady-State 65 CU/day)
# ---------------------------------------------------------------------------

@dataclass
class CuGovernorConfig:
    floor_cu: float = 500.0
    steady_daily_cap_cu: float = 65.0
    hard_stop_cu: float = 50.0
    max_total_slots: int = 7
    max_gpu_slots: int = 3
    max_tpu_slots: int = 2
    max_cpu_slots: int = 4


@dataclass
class CuDecision:
    mode: str  # "sprint" | "steady_state" | "halted_daily_cap" | "halted_hard_stop" | "halted_unknown_balance"
    admitted: bool
    max_concurrent_slots: int
    remaining_budget_today_cu: float
    reason: str


class CuGovernor:
    """Two-stage CU budget governor with sprint-to-floor and steady-state daily cap."""

    def __init__(self, cfg: CuGovernorConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        *,
        balance_cu: Optional[float],
        spent_today_cu: float,
        pending_jobs: int,
        steady_work_admitted: bool = True,
    ) -> CuDecision:
        if balance_cu is None:
            return CuDecision("halted_unknown_balance", False, 0, 0.0, "colab_usage_unavailable")
        if balance_cu <= self.cfg.hard_stop_cu:
            return CuDecision(
                "halted_hard_stop",
                False,
                0,
                0.0,
                f"balance {balance_cu:.2f} <= hard_stop_cu {self.cfg.hard_stop_cu:.2f}",
            )
        if pending_jobs <= 0:
            return CuDecision(
                "sprint" if balance_cu > self.cfg.floor_cu else "steady_state",
                False,
                0,
                max(0.0, balance_cu - self.cfg.floor_cu) if balance_cu > self.cfg.floor_cu else max(0.0, self.cfg.steady_daily_cap_cu - spent_today_cu),
                "idle_no_pending_jobs",
            )

        if balance_cu > self.cfg.floor_cu:
            avail = round(balance_cu - self.cfg.floor_cu, 2)
            return CuDecision(
                mode="sprint",
                admitted=True,
                max_concurrent_slots=self.cfg.max_total_slots,
                remaining_budget_today_cu=avail,
                reason=f"sprint_burn_above_floor (balance={balance_cu:.2f}, floor={self.cfg.floor_cu:.2f}, headroom={avail:.2f} CU)",
            )

        # Steady-state mode (balance_cu <= floor_cu): enforce daily cap and work-admission gate
        if not steady_work_admitted:
            return CuDecision(
                mode="steady_state",
                admitted=False,
                max_concurrent_slots=0,
                remaining_budget_today_cu=max(0.0, round(self.cfg.steady_daily_cap_cu - spent_today_cu, 2)),
                reason="steady_state_idle_no_new_work_admitted",
            )

        rem_today = round(self.cfg.steady_daily_cap_cu - spent_today_cu, 2)
        if rem_today <= 0.5:
            return CuDecision(
                mode="halted_daily_cap",
                admitted=False,
                max_concurrent_slots=0,
                remaining_budget_today_cu=0.0,
                reason=f"steady_daily_cap_reached (spent_today={spent_today_cu:.2f} >= {self.cfg.steady_daily_cap_cu:.2f})",
            )

        # Throttle concurrency in steady state to at most 2 slots so we don't overshoot daily cap
        steady_slots = min(2, self.cfg.max_total_slots)
        return CuDecision(
            mode="steady_state",
            admitted=True,
            max_concurrent_slots=steady_slots,
            remaining_budget_today_cu=rem_today,
            reason=f"steady_state_admitted (balance={balance_cu:.2f} <= {self.cfg.floor_cu:.2f}, rem_today={rem_today:.2f} CU)",
        )


# ---------------------------------------------------------------------------
# Dual Pluggable Sync Backends (`--sync local` and `--sync s3`)
# ---------------------------------------------------------------------------

class SyncBackend:
    """Abstract artifact sync backend (local Colab upload/download or S3-compatible R2/GCS)."""

    mode: str = "abstract"

    def stage_inputs(self, session: str, uploads: Sequence[Tuple[Path, str]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def collect_outputs(self, session: str, downloads: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"mode": self.mode}


class LocalColabSyncBackend(SyncBackend):
    """Default zero-cloud-key sync via `colab upload` and `colab download` to local SSD."""

    mode = "local"

    def __init__(
        self,
        colab_runner: Callable[[Sequence[str], int], subprocess.CompletedProcess[str]],
        local_root: Path,
    ) -> None:
        self._run = colab_runner
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def stage_inputs(self, session: str, uploads: Sequence[Tuple[Path, str]]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for local_path, remote_path in uploads:
            assert_public_path(str(local_path))
            if not local_path.is_file():
                raise FileNotFoundError(f"Upload file missing: {local_path}")
            p = self._run(["upload", "-s", session, str(local_path), remote_path], 300)
            recs.append({
                "direction": "upload",
                "local": str(local_path),
                "remote": remote_path,
                "size_bytes": local_path.stat().st_size,
                "rc": p.returncode,
            })
            if p.returncode != 0:
                raise RuntimeError(f"colab upload failed (rc={p.returncode}) for {local_path.name}")
        return recs

    def collect_outputs(self, session: str, downloads: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for remote_path, local_dest in downloads:
            local_dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            p = self._run(["download", "-s", session, remote_path, str(local_dest)], 300)
            ok = p.returncode == 0 and local_dest.is_file()
            recs.append({
                "direction": "download",
                "remote": remote_path,
                "local": str(local_dest),
                "size_bytes": local_dest.stat().st_size if ok else 0,
                "rc": p.returncode,
                "ok": ok,
            })
        return recs

    def describe(self) -> Dict[str, Any]:
        return {"mode": "local", "local_root": str(self.local_root), "cloud_keys_on_colab": False}


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = "auto"
    prefix: str = "freetpu-fleet"

    @classmethod
    def from_env(cls) -> "S3Config":
        endpoint = os.environ.get("FREETPU_S3_ENDPOINT", "").strip().rstrip("/")
        bucket = os.environ.get("FREETPU_S3_BUCKET", "").strip()
        ak = os.environ.get("FREETPU_S3_ACCESS_KEY_ID", "").strip()
        sk = os.environ.get("FREETPU_S3_SECRET_ACCESS_KEY", "").strip()
        region = os.environ.get("FREETPU_S3_REGION", "auto").strip() or "auto"
        prefix = os.environ.get("FREETPU_S3_PREFIX", "freetpu-fleet").strip().strip("/")
        missing = [
            k for k, v in (
                ("FREETPU_S3_ENDPOINT", endpoint),
                ("FREETPU_S3_BUCKET", bucket),
                ("FREETPU_S3_ACCESS_KEY_ID", ak),
                ("FREETPU_S3_SECRET_ACCESS_KEY", sk),
            ) if not v
        ]
        if missing:
            raise ValueError(f"Missing required S3/R2 env vars for --sync s3: {', '.join(missing)}")
        return cls(
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=ak,
            secret_access_key=sk,
            region=region,
            prefix=prefix,
        )


def sign_s3_sigv4(
    *,
    method: str,
    url: str,
    payload: bytes,
    access_key_id: str,
    secret_access_key: str,
    region: str = "auto",
    amz_date: Optional[str] = None,
    content_type: str = "application/octet-stream",
) -> Dict[str, str]:
    """Pure-stdlib AWS Signature Version 4 signer for S3-compatible endpoints (R2/GCS/MinIO/S3)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    canonical_uri = urllib.parse.quote(parsed.path or "/", safe="/-_.~")
    canonical_qs = parsed.query
    now_dt = datetime.now(timezone.utc)
    amz_dt = amz_date or now_dt.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_dt[:8]
    payload_hash = hashlib.sha256(payload).hexdigest()

    headers_to_sign = {
        "content-type": content_type,
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_dt,
    }
    signed_keys = sorted(headers_to_sign.keys())
    canonical_headers = "".join(f"{k}:{headers_to_sign[k]}\n" for k in signed_keys)
    signed_headers = ";".join(signed_keys)

    canonical_req = "\n".join([
        method.upper(),
        canonical_uri,
        canonical_qs,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_dt,
        scope,
        hashlib.sha256(canonical_req.encode("utf-8")).hexdigest(),
    ])

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _hmac(("AWS4" + secret_access_key).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, "s3")
    k_signing = _hmac(k_service, "aws4_request")
    sig = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "Content-Type": content_type,
        "Host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_dt,
        "Authorization": f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, SignedHeaders={signed_headers}, Signature={sig}",
    }


class S3CompatibleSyncBackend(SyncBackend):
    """Generic S3-compatible sync (Cloudflare R2 / GCS HMAC / MinIO / S3) with zero secret logging."""

    mode = "s3"

    def __init__(
        self,
        cfg: S3Config,
        local_fallback: Optional[LocalColabSyncBackend] = None,
        http_opener: Optional[Callable[[urllib.request.Request, int], Tuple[int, bytes]]] = None,
    ) -> None:
        self.cfg = cfg
        self.local_fallback = local_fallback
        self._http_opener = http_opener or self._default_http

    @staticmethod
    def _default_http(req: urllib.request.Request, timeout_s: int) -> Tuple[int, bytes]:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read()

    def object_url(self, key: str) -> str:
        clean_key = key.lstrip("/")
        full_key = f"{self.cfg.prefix}/{clean_key}" if self.cfg.prefix else clean_key
        return f"{self.cfg.endpoint}/{self.cfg.bucket}/{full_key}"

    def put_bytes(self, key: str, data: bytes) -> Dict[str, Any]:
        url = self.object_url(key)
        hdrs = sign_s3_sigv4(
            method="PUT",
            url=url,
            payload=data,
            access_key_id=self.cfg.access_key_id,
            secret_access_key=self.cfg.secret_access_key,
            region=self.cfg.region,
        )
        req = urllib.request.Request(url, data=data, headers=hdrs, method="PUT")
        status, _ = self._http_opener(req, 120)
        if status not in (200, 201, 204):
            raise RuntimeError(f"S3 PUT failed with HTTP {status} for key={key}")
        return {"key": key, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": status}

    def get_bytes(self, key: str) -> bytes:
        url = self.object_url(key)
        hdrs = sign_s3_sigv4(
            method="GET",
            url=url,
            payload=b"",
            access_key_id=self.cfg.access_key_id,
            secret_access_key=self.cfg.secret_access_key,
            region=self.cfg.region,
        )
        req = urllib.request.Request(url, headers=hdrs, method="GET")
        status, body = self._http_opener(req, 120)
        if status != 200:
            raise RuntimeError(f"S3 GET failed with HTTP {status} for key={key}")
        return body

    def stage_inputs(self, session: str, uploads: Sequence[Tuple[Path, str]]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for local_path, remote_path in uploads:
            assert_public_path(str(local_path))
            data = local_path.read_bytes()
            key = f"sessions/{session}/in/{local_path.name}"
            put_rec = self.put_bytes(key, data)
            recs.append({"direction": "s3_put", "remote_target": remote_path, **put_rec})
        if self.local_fallback is not None:
            recs.extend(self.local_fallback.stage_inputs(session, uploads))
        return recs

    def collect_outputs(self, session: str, downloads: Sequence[Tuple[str, Path]]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        if self.local_fallback is not None:
            local_recs = self.local_fallback.collect_outputs(session, downloads)
            recs.extend(local_recs)
            for item in local_recs:
                if item.get("ok"):
                    p = Path(item["local"])
                    key = f"sessions/{session}/out/{p.name}"
                    recs.append({"direction": "s3_archive", **self.put_bytes(key, p.read_bytes())})
        return recs

    def describe(self) -> Dict[str, Any]:
        # Strictly never emit access_key_id or secret_access_key!
        return {
            "mode": "s3",
            "endpoint": self.cfg.endpoint,
            "bucket": self.cfg.bucket,
            "region": self.cfg.region,
            "prefix": self.cfg.prefix,
            "credentials_logged": False,
        }


# ---------------------------------------------------------------------------
# Fleet Job Spec, Colab Control Plane Wrapper & 7-Slot Dispatcher
# ---------------------------------------------------------------------------

@dataclass
class FleetJob:
    job_id: str
    wave: int
    accelerator_class: str  # "GPU" | "TPU" | "CPU"
    script: Path
    sku_preference: Tuple[str, ...] = ()
    uploads: Tuple[Tuple[Path, str], ...] = ()
    downloads: Tuple[Tuple[str, Path], ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    timeout_s: int = 1800
    max_cu: float = 40.0
    description: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], base_dir: Path) -> "FleetJob":
        acc = str(raw.get("accelerator_class", "GPU")).upper()
        if acc not in ("GPU", "TPU", "CPU"):
            raise ValueError(f"Unsupported accelerator_class={acc!r}")
        script_p = Path(raw["script"])
        if not script_p.is_absolute():
            script_p = (base_dir / script_p).resolve()
        assert_public_path(str(script_p))
        uploads: List[Tuple[Path, str]] = []
        for u in raw.get("uploads", []):
            lp = Path(u["local"])
            if not lp.is_absolute():
                lp = (base_dir / lp).resolve()
            assert_public_path(str(lp))
            uploads.append((lp, str(u["remote"])))
        downloads: List[Tuple[str, Path]] = []
        for d in raw.get("downloads", []):
            lp = Path(d["local"])
            if not lp.is_absolute():
                lp = (base_dir / lp).resolve()
            downloads.append((str(d["remote"]), lp))
        default_pref = (
            DEFAULT_GPU_FALLBACK if acc == "GPU"
            else (DEFAULT_TPU_FALLBACK if acc == "TPU" else DEFAULT_CPU_FALLBACK)
        )
        pref = tuple(str(x).upper() for x in raw.get("sku_preference", default_pref))
        return cls(
            job_id=str(raw["job_id"]),
            wave=int(raw.get("wave", 1)),
            accelerator_class=acc,
            script=script_p,
            sku_preference=pref,
            uploads=tuple(uploads),
            downloads=tuple(downloads),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            timeout_s=int(raw.get("timeout_s", 1800)),
            max_cu=float(raw.get("max_cu", 40.0)),
            description=str(raw.get("description", "")),
        )


def parse_sessions_output(stdout: str) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    for raw in stdout.splitlines():
        m = SESSION_LINE_RE.match(raw.strip())
        if not m:
            continue
        ep = m.group("endpoint").strip()
        zm = ZONE_TOKEN_RE.search(ep)
        variant = m.group("variant").strip().upper()
        kind = "GPU" if variant == "GPU" else ("TPU" if variant == "TPU" else "CPU")
        sessions.append({
            "name": m.group("name").strip(),
            "endpoint": ep,
            "zone": zm.group("zone") if zm else None,
            "hardware": m.group("hw").strip(),
            "shape": m.group("shape").strip(),
            "variant": variant,
            "kind": kind,
        })
    return sessions


def parse_usage_output(text: str) -> Dict[str, Any]:
    bal_m = re.search(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)", text)
    rate_m = re.search(r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)", text)
    act_m = re.search(r"Active assignments:\s*([0-9]+)", text)
    return {
        "balance_cu": float(bal_m.group(1)) if bal_m else None,
        "usage_rate_hr": float(rate_m.group(1)) if rate_m else None,
        "active_assignments": int(act_m.group(1)) if act_m else None,
    }


def classify_assign_error(text: str) -> str:
    low = text.lower()
    if "precondition failed" in low or "too many active" in low or "toomanyassignmentserror" in low or " 412" in low:
        return "412_TOO_MANY_ACTIVE"
    if "service unavailable" in low or " 503" in low:
        return "503_UNAVAILABLE"
    if "too many requests" in low or " 429" in low:
        return "429_RATE_LIMITED"
    if "readtimeout" in low or "read timed out" in low:
        return "READ_TIMEOUT"
    return "OTHER_ERROR"


class ColabFleetDispatcher:
    """Thread-safe 7-slot Colab Fleet dispatcher with durable telemetry and isolated teardown."""

    def __init__(
        self,
        *,
        colab_bin: str,
        auth: str,
        run_dir: Path,
        governor_cfg: CuGovernorConfig,
        sync_mode: str = "local",
        s3_cfg: Optional[S3Config] = None,
        sku_cooldown_s: float = 90.0,
        command_runner: Optional[Callable[[Sequence[str], int], subprocess.CompletedProcess[str]]] = None,
    ) -> None:
        self.colab_bin = colab_bin
        self.auth = auth
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.run_dir, 0o700)
        self.cfg_dir = self.run_dir / "configs"
        self.cfg_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.list_cfg = self.cfg_dir / "list-sessions.json"
        self.governor = CuGovernor(governor_cfg)
        self.sku_cooldown_s = sku_cooldown_s
        self._lock = threading.Lock()
        self._owned_sessions: Dict[str, Dict[str, Any]] = {}
        self._sku_cooldown_until: Dict[str, float] = {}
        self._job_states: Dict[str, Dict[str, Any]] = {}
        self._spent_today_cu: float = 0.0
        self._stop_requested = threading.Event()
        self._custom_runner = command_runner

        local_sync = LocalColabSyncBackend(self._run_colab_shared, self.run_dir / "artifacts")
        if sync_mode == "s3":
            resolved_s3 = s3_cfg or S3Config.from_env()
            self.sync: SyncBackend = S3CompatibleSyncBackend(resolved_s3, local_fallback=local_sync)
        else:
            self.sync = local_sync

    def _run_colab_with_cfg(
        self,
        cfg_path: Path,
        args: Sequence[str],
        timeout_s: int = 120,
        request_timeout_s: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        if self._custom_runner is not None:
            return self._custom_runner(list(args), timeout_s)
        env = dict(os.environ)
        env["COLAB_CLI_ASSIGN_RETRIES"] = "0"
        env["COLAB_CLI_REQUEST_TIMEOUT"] = str(request_timeout_s)
        cmd = [self.colab_bin, f"--auth={self.auth}", "--config", str(cfg_path), *args]
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )

    def _run_colab_shared(self, args: Sequence[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
        # Route per-session upload/download to that session's isolated config when present
        sess_name = None
        for i, tok in enumerate(args):
            if tok in ("-s", "--session") and i + 1 < len(args):
                sess_name = args[i + 1]
                break
        cfg = self.cfg_dir / f"{sess_name}.json" if sess_name else self.list_cfg
        return self._run_colab_with_cfg(cfg, args, timeout_s=timeout_s)

    def get_usage(self) -> Dict[str, Any]:
        p = self._run_colab_with_cfg(self.list_cfg, ["usage"], timeout_s=45, request_timeout_s=30)
        return parse_usage_output(p.stdout + "\n" + p.stderr)

    def list_sessions(self) -> List[Dict[str, Any]]:
        p = self._run_colab_with_cfg(self.list_cfg, ["sessions"], timeout_s=45, request_timeout_s=30)
        return parse_sessions_output(p.stdout)

    def log_event(self, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"timestamp_utc": utc_now_iso(), "event": kind, **payload}
        line = redact_secrets(json.dumps(rec, sort_keys=True))
        with self._lock:
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            (self.run_dir / "heartbeat").write_text(utc_now_iso() + "\n", encoding="utf-8")

    def write_state_md(self, usage: Dict[str, Any], decision: CuDecision) -> None:
        with self._lock:
            lines = [
                "# Colab 7-Slot Fleet State",
                "",
                f"- **Updated (UTC):** `{utc_now_iso()}`",
                f"- **Governor Mode:** `{decision.mode}` (`admitted={decision.admitted}`, reason: `{decision.reason}`)",
                f"- **CU Balance:** `{usage.get('balance_cu')}` CUs (`rate={usage.get('usage_rate_hr')}/hr`, `active={usage.get('active_assignments')}`)",
                f"- **Sync Backend:** `{self.sync.describe()}`",
                f"- **Owned Active Sessions:** `{len(self._owned_sessions)}` (`{sorted(self._owned_sessions.keys())}`)",
                "",
                "| Job ID | Wave | Class | Status | SKU | Zone | Elapsed (s) | Notes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for jid, st in sorted(self._job_states.items()):
                lines.append(
                    f"| `{jid}` | {st.get('wave')} | `{st.get('class')}` | `{st.get('status')}` | "
                    f"`{st.get('sku') or '-'}` | `{st.get('zone') or '-'}` | {st.get('elapsed_s', '-')} | {st.get('note', '')} |"
                )
            (self.run_dir / "STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _can_admit_slot(self, acc_class: str, active_sessions: List[Dict[str, Any]], max_slots: int) -> Tuple[bool, str]:
        total_active = len(active_sessions)
        if total_active >= max_slots:
            return False, f"total_cap_reached ({total_active}/{max_slots})"
        gpu_cnt = sum(1 for s in active_sessions if s["kind"] == "GPU")
        tpu_cnt = sum(1 for s in active_sessions if s["kind"] == "TPU")
        cpu_cnt = sum(1 for s in active_sessions if s["kind"] == "CPU")
        if acc_class == "GPU" and gpu_cnt >= self.governor.cfg.max_gpu_slots:
            return False, f"gpu_cap_reached ({gpu_cnt}/{self.governor.cfg.max_gpu_slots})"
        if acc_class == "TPU" and tpu_cnt >= self.governor.cfg.max_tpu_slots:
            return False, f"tpu_cap_reached ({tpu_cnt}/{self.governor.cfg.max_tpu_slots})"
        if acc_class == "CPU" and cpu_cnt >= self.governor.cfg.max_cpu_slots:
            return False, f"cpu_cap_reached ({cpu_cnt}/{self.governor.cfg.max_cpu_slots})"
        return True, "ok"

    def _provision_job_session(self, job: FleetJob) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], str]:
        now = time.monotonic()
        for sku in job.sku_preference:
            if sku not in SKU_CLI_FLAGS:
                continue
            if self._sku_cooldown_until.get(sku, 0.0) > now:
                continue
            sess_name = f"fleet-{job.job_id[:18]}-{sku.lower()}-{uuid.uuid4().hex[:6]}"
            cfg_path = self.cfg_dir / f"{sess_name}.json"
            flags = list(SKU_CLI_FLAGS[sku])
            req_to = 150 if SKU_KIND[sku] == "TPU" else 90
            before_eps = {s["endpoint"] for s in self.list_sessions()}
            p = self._run_colab_with_cfg(
                cfg_path,
                ["new", "-s", sess_name, *flags],
                timeout_s=req_to + 45,
                request_timeout_s=req_to,
            )
            if p.returncode == 0:
                after = self.list_sessions()
                matched = [s for s in after if s["name"] == sess_name]
                if not matched:
                    new_eps = [s for s in after if s["endpoint"] not in before_eps]
                    matched = new_eps[:1]
                assign = matched[0] if matched else {"name": sess_name, "endpoint": "unknown", "zone": None}
                with self._lock:
                    self._owned_sessions[sess_name] = {
                        "job_id": job.job_id,
                        "sku": sku,
                        "cfg_path": str(cfg_path),
                        "assignment": assign,
                        "started_utc": utc_now_iso(),
                    }
                self.log_event("session_allocated", {"job_id": job.job_id, "session": sess_name, "sku": sku, "zone": assign.get("zone")})
                return sess_name, sku, assign, "ok"

            err_code = classify_assign_error(p.stdout + "\n" + p.stderr)
            self.log_event("sku_refused", {"job_id": job.job_id, "sku": sku, "error_code": err_code})
            if err_code in ("503_UNAVAILABLE", "429_RATE_LIMITED", "READ_TIMEOUT"):
                self._sku_cooldown_until[sku] = time.monotonic() + self.sku_cooldown_s
                # On ReadTimeout, reconcile before moving to next SKU so we never double-allocate!
                if err_code == "READ_TIMEOUT":
                    after = self.list_sessions()
                    leaked = [s for s in after if s["name"] == sess_name]
                    if leaked:
                        self.stop_owned_session(sess_name)
                continue
            if err_code == "412_TOO_MANY_ACTIVE":
                return None, None, None, err_code
        return None, None, None, "no_sku_available"

    def stop_owned_session(self, sess_name: str) -> Dict[str, Any]:
        if not sess_name.startswith("fleet-"):
            raise RuntimeError(f"SAFETY REFUSAL: refusing to stop non-fleet session {sess_name!r}")
        cfg_path = self.cfg_dir / f"{sess_name}.json"
        p = self._run_colab_with_cfg(cfg_path, ["stop", "-s", sess_name], timeout_s=60, request_timeout_s=45)
        with self._lock:
            self._owned_sessions.pop(sess_name, None)
        self.log_event("session_stopped", {"session": sess_name, "rc": p.returncode})
        return {"session": sess_name, "rc": p.returncode}

    def teardown_all_owned(self) -> Dict[str, Any]:
        with self._lock:
            names = list(self._owned_sessions.keys())
        stopped = []
        for n in reversed(names):
            try:
                stopped.append(self.stop_owned_session(n))
            except Exception as exc:
                stopped.append({"session": n, "rc": -1, "error": str(exc)})
        after = self.list_sessions()
        lingering = [s for s in after if s["name"].startswith("fleet-")]
        self.log_event("teardown_verified", {"stopped_count": len(stopped), "lingering_fleet_count": len(lingering)})
        return {
            "stopped": stopped,
            "lingering_fleet_sessions": lingering,
            "zero_fleet_leaked": len(lingering) == 0,
            "external_sessions_preserved": [s["endpoint"] for s in after if not s["name"].startswith("fleet-")],
        }

    def _execute_one_job(self, job: FleetJob, sess_name: str, sku: str, assign: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cfg_path = self.cfg_dir / f"{sess_name}.json"
        log_path = self.run_dir / f"{job.job_id}.log"
        rc = 1
        note = ""
        try:
            if job.uploads:
                self.sync.stage_inputs(sess_name, job.uploads)
            exec_args = ["exec", "-s", sess_name, "--timeout", str(job.timeout_s)]
            for k, v in sorted(job.env.items()):
                exec_args.extend(["--env", f"{k}={v}"])
            exec_args.extend(["-f", str(job.script)])
            p = self._run_colab_with_cfg(cfg_path, exec_args, timeout_s=job.timeout_s + 120, request_timeout_s=120)
            rc = p.returncode
            log_path.write_text(redact_secrets((p.stdout or "") + "\n" + (p.stderr or "")), encoding="utf-8")
            if job.downloads:
                self.sync.collect_outputs(sess_name, job.downloads)
            note = "ok" if rc == 0 else f"exec_rc_{rc}"
        except Exception as exc:
            note = redact_secrets(f"{type(exc).__name__}: {exc}")[:200]
            rc = 1
        finally:
            self.stop_owned_session(sess_name)
        elapsed_s = round(time.perf_counter() - t0, 2)
        est_cu = round((elapsed_s / 3600.0) * SKU_EST_CU_HR.get(sku, 1.0), 4)
        with self._lock:
            self._spent_today_cu += est_cu
            self._job_states[job.job_id] = {
                "wave": job.wave,
                "class": job.accelerator_class,
                "status": "done" if rc == 0 else "failed",
                "sku": sku,
                "zone": assign.get("zone"),
                "elapsed_s": elapsed_s,
                "est_cu": est_cu,
                "rc": rc,
                "log": str(log_path),
                "note": note,
            }
        self.log_event("job_finished", {"job_id": job.job_id, "rc": rc, "sku": sku, "elapsed_s": elapsed_s, "est_cu": est_cu})
        return self._job_states[job.job_id]

    def plan_dry_run(self, jobs: Sequence[FleetJob]) -> Dict[str, Any]:
        usage = self.get_usage() if self._custom_runner is not None else {"balance_cu": 7323.47, "usage_rate_hr": 0.0, "active_assignments": 0}
        dec = self.governor.evaluate(
            balance_cu=usage.get("balance_cu"),
            spent_today_cu=0.0,
            pending_jobs=len(jobs),
        )
        return {
            "status": "dry_run",
            "timestamp_utc": utc_now_iso(),
            "governor": dataclasses.asdict(dec),
            "governor_config": dataclasses.asdict(self.governor.cfg),
            "sync_backend": self.sync.describe(),
            "jobs_planned": [
                {
                    "job_id": j.job_id,
                    "wave": j.wave,
                    "accelerator_class": j.accelerator_class,
                    "sku_preference": list(j.sku_preference),
                    "script": str(j.script),
                    "timeout_s": j.timeout_s,
                    "max_cu": j.max_cu,
                }
                for j in jobs
            ],
        }

    def run_queue(self, jobs: Sequence[FleetJob]) -> Dict[str, Any]:
        """Execute jobs wave-by-wave across up to 7 concurrent slots with CU governor & guaranteed teardown."""
        def _sig(signum, _frame):
            self._stop_requested.set()
            self.log_event("signal_received", {"signal": signum})

        old_int = signal.signal(signal.SIGINT, _sig)
        old_term = signal.signal(signal.SIGTERM, _sig)
        usage_start = self.get_usage()
        for j in jobs:
            self._job_states[j.job_id] = {
                "wave": j.wave,
                "class": j.accelerator_class,
                "status": "pending",
                "sku": None,
                "zone": None,
            }

        pending = sorted(jobs, key=lambda x: (x.wave, x.job_id))
        running: Dict[concurrent.futures.Future[Dict[str, Any]], FleetJob] = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.governor.cfg.max_total_slots) as pool:
                while (pending or running) and not self._stop_requested.is_set():
                    # Reap completed futures
                    done_futs = [f for f in list(running.keys()) if f.done()]
                    for f in done_futs:
                        running.pop(f, None)

                    usage = self.get_usage()
                    dec = self.governor.evaluate(
                        balance_cu=usage.get("balance_cu"),
                        spent_today_cu=self._spent_today_cu,
                        pending_jobs=len(pending),
                    )
                    self.write_state_md(usage, dec)

                    if not dec.admitted and not running:
                        self.log_event("governor_halt", dataclasses.asdict(dec))
                        break

                    if dec.admitted and pending:
                        # Admit jobs from the lowest active wave (or concurrent TPU/CPU lanes)
                        active_sessions = self.list_sessions()
                        launched_any = False
                        next_pending: List[FleetJob] = []
                        for job in pending:
                            ok_slot, _reason = self._can_admit_slot(
                                job.accelerator_class,
                                active_sessions,
                                dec.max_concurrent_slots,
                            )
                            if not ok_slot:
                                next_pending.append(job)
                                continue
                            sess_name, sku, assign, status = self._provision_job_session(job)
                            if sess_name and sku and assign:
                                with self._lock:
                                    self._job_states[job.job_id]["status"] = "running"
                                    self._job_states[job.job_id]["sku"] = sku
                                    self._job_states[job.job_id]["zone"] = assign.get("zone")
                                active_sessions.append({
                                    "name": sess_name,
                                    "endpoint": assign.get("endpoint", ""),
                                    "kind": job.accelerator_class,
                                })
                                fut = pool.submit(self._execute_one_job, job, sess_name, sku, assign)
                                running[fut] = job
                                launched_any = True
                            else:
                                with self._lock:
                                    self._job_states[job.job_id]["note"] = f"deferred:{status}"
                                next_pending.append(job)
                                if status == "412_TOO_MANY_ACTIVE":
                                    # Stop admitting more slots right now
                                    next_pending.extend(pending[pending.index(job) + 1 :])
                                    break
                        pending = next_pending
                        if not launched_any and not running:
                            # All SKUs on cooldown or refused; stop cleanly instead of spinning
                            break

                    if running:
                        concurrent.futures.wait(list(running.keys()), timeout=5.0, return_when=concurrent.futures.FIRST_COMPLETED)
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            teardown = self.teardown_all_owned()
            usage_end = self.get_usage()
            cu_delta = (
                round(usage_start["balance_cu"] - usage_end["balance_cu"], 4)
                if usage_start.get("balance_cu") is not None and usage_end.get("balance_cu") is not None
                else None
            )
            proof = {
                "schema": "freetpu.colab_fleet_proof.v1",
                "timestamp_utc": utc_now_iso(),
                "usage_before": usage_start,
                "usage_after": usage_end,
                "cu_delta": cu_delta,
                "sync_backend": self.sync.describe(),
                "jobs": self._job_states,
                "teardown": teardown,
            }
            (self.run_dir / "fleet_proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
            self._write_proof_md(proof)
        return proof

    def _write_proof_md(self, proof: Dict[str, Any]) -> None:
        lines = [
            "# Colab 7-Slot Fleet Proof",
            "",
            f"- **Completed (UTC):** `{proof['timestamp_utc']}`",
            f"- **CU Telemetry:** `{proof['usage_before'].get('balance_cu')} -> {proof['usage_after'].get('balance_cu')}` (delta: `{proof['cu_delta']} CUs`)",
            f"- **Teardown Verified (`0` fleet-* leaked):** `{proof['teardown']['zero_fleet_leaked']}`",
            f"- **External Sessions Preserved:** `{proof['teardown']['external_sessions_preserved']}`",
            f"- **Sync Backend:** `{proof['sync_backend']}`",
            "",
            "| Job ID | Wave | Class | Status | SKU | Zone | Elapsed (s) | Est CU |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for jid, st in sorted(proof["jobs"].items()):
            lines.append(
                f"| `{jid}` | {st.get('wave')} | `{st.get('class')}` | `{st.get('status')}` | "
                f"`{st.get('sku') or '-'}` | `{st.get('zone') or '-'}` | {st.get('elapsed_s', '-')} | {st.get('est_cu', '-')} |"
            )
        (self.run_dir / "PROOF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Offline Self-Test (0 CUs, 0 Network)
# ---------------------------------------------------------------------------

def run_self_test() -> Dict[str, Any]:
    """Comprehensive offline self-test exercising CU governor, 7-slot scheduler, dual sync, and isolation."""
    # 1. Two-stage CU governor
    gov = CuGovernor(CuGovernorConfig(floor_cu=500.0, steady_daily_cap_cu=65.0, hard_stop_cu=50.0))
    d_sprint = gov.evaluate(balance_cu=7323.47, spent_today_cu=120.0, pending_jobs=5)
    assert d_sprint.mode == "sprint" and d_sprint.admitted and d_sprint.max_concurrent_slots == 7
    d_steady = gov.evaluate(balance_cu=480.0, spent_today_cu=20.0, pending_jobs=3, steady_work_admitted=True)
    assert d_steady.mode == "steady_state" and d_steady.admitted and d_steady.max_concurrent_slots == 2
    d_cap = gov.evaluate(balance_cu=480.0, spent_today_cu=65.0, pending_jobs=3, steady_work_admitted=True)
    assert d_cap.mode == "halted_daily_cap" and not d_cap.admitted
    d_idle = gov.evaluate(balance_cu=480.0, spent_today_cu=0.0, pending_jobs=3, steady_work_admitted=False)
    assert not d_idle.admitted
    d_stop = gov.evaluate(balance_cu=45.0, spent_today_cu=0.0, pending_jobs=3)
    assert d_stop.mode == "halted_hard_stop" and not d_stop.admitted

    # 2. S3 SigV4 deterministic signing & secret redaction
    hdrs = sign_s3_sigv4(
        method="PUT",
        url="https://example-r2.invalid/test-bucket/fleet/a.bin",
        payload=b"hello-freetpu",
        access_key_id="TESTKEY123",
        secret_access_key="TESTSECRET456",
        region="auto",
        amz_date="20261126T180000Z",
    )
    assert hdrs["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=TESTKEY123/20261126/auto/s3/aws4_request")
    assert "TESTSECRET456" not in hdrs["Authorization"]
    assert "REDACTED" in redact_secrets("Authorization: Bearer abc123secret")

    # 3. Mocked 7-slot run with 1 pre-existing external session (must stay untouched!) + 503 H100 -> A100 fallback
    with tempfile.TemporaryDirectory(prefix="freetpu_fleet_st_") as td:
        td_p = Path(td)
        dummy_script = td_p / "worker.py"
        dummy_script.write_text("print('ok')\n", encoding="utf-8")
        mock_sessions: Dict[str, str] = {
            "?": "[?] gpu-a100-s-usc1f0-external | Hardware: A100 | Shape: Standard | Variant: GPU"
        }
        lock = threading.Lock()
        stopped_names: List[str] = []

        def mock_runner(args: Sequence[str], _timeout: int) -> subprocess.CompletedProcess[str]:
            cmd = args[0]
            with lock:
                if cmd == "usage":
                    return subprocess.CompletedProcess(
                        args, 0,
                        f"Current balance: 7320.00 compute units\nUsage rate: 8.16/hr\nActive assignments: {len(mock_sessions)}\n",
                        "",
                    )
                if cmd == "sessions":
                    return subprocess.CompletedProcess(args, 0, "\n".join(mock_sessions.values()) + "\n", "")
                if cmd == "new":
                    sname = args[args.index("-s") + 1]
                    if "--gpu" in args and args[args.index("--gpu") + 1] == "H100":
                        return subprocess.CompletedProcess(args, 1, "", "503 Service Unavailable")
                    if "--gpu" in args:
                        hw = args[args.index("--gpu") + 1]
                        mock_sessions[sname] = f"[{sname}] gpu-{hw.lower()}-s-usc1a0-m1 | Hardware: {hw} | Shape: Standard | Variant: GPU"
                    elif "--tpu" in args:
                        hw = args[args.index("--tpu") + 1]
                        mock_sessions[sname] = f"[{sname}] tpu-{hw.lower()}-s-usw4a0-m1 | Hardware: {hw} | Shape: Standard | Variant: TPU"
                    else:
                        mock_sessions[sname] = f"[{sname}] m-hm-ase1a0-m1 | Hardware: CPU | Shape: High-RAM | Variant: DEFAULT"
                    return subprocess.CompletedProcess(args, 0, "Session created\n", "")
                if cmd == "stop":
                    sname = args[args.index("-s") + 1]
                    stopped_names.append(sname)
                    mock_sessions.pop(sname, None)
                    return subprocess.CompletedProcess(args, 0, "Stopped\n", "")
                if cmd == "exec":
                    return subprocess.CompletedProcess(args, 0, "ok\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        disp = ColabFleetDispatcher(
            colab_bin="colab",
            auth="adc",
            run_dir=td_p / "run",
            governor_cfg=CuGovernorConfig(floor_cu=500.0, steady_daily_cap_cu=65.0),
            sync_mode="local",
            command_runner=mock_runner,
        )
        jobs = [
            FleetJob("gpu_lane1", 1, "GPU", dummy_script, sku_preference=("H100", "A100")),
            FleetJob("tpu_lane1", 1, "TPU", dummy_script, sku_preference=("V5E1",)),
            FleetJob("cpu_twin1", 1, "CPU", dummy_script, sku_preference=("CPU_HM",)),
        ]
        proof = disp.run_queue(jobs)
        assert proof["teardown"]["zero_fleet_leaked"] is True
        assert proof["teardown"]["external_sessions_preserved"] == ["gpu-a100-s-usc1f0-external"]
        assert proof["jobs"]["gpu_lane1"]["sku"] == "A100"  # fell back cleanly from 503 H100
        assert proof["jobs"]["tpu_lane1"]["status"] == "done"
        assert proof["jobs"]["cpu_twin1"]["status"] == "done"
        assert all(n.startswith("fleet-") for n in stopped_names)
        assert (td_p / "run" / "STATE.md").is_file()
        assert (td_p / "run" / "PROOF.md").is_file()
        assert (td_p / "run" / "events.jsonl").is_file()
        assert (td_p / "run" / "heartbeat").is_file()

    return {
        "schema": "freetpu.colab_fleet_selftest.v1",
        "all_passed": True,
        "checks": [
            "two_stage_cu_governor",
            "s3_sigv4_and_redaction",
            "7slot_scheduler_fallback_and_external_isolation",
            "durable_state_proof_events_heartbeat",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean-room 7-slot Google Colab Fleet Dispatcher (GPU+TPU+CPU) with Dual Sync & Two-Stage CU Governor."
    )
    parser.add_argument("--colab-bin", default="colab", help="Path to google-colab-cli executable (default: colab).")
    parser.add_argument("--auth", choices=("adc", "oauth2"), default="adc", help="Colab CLI auth mode (default: adc).")
    parser.add_argument("--queue", type=Path, help="External JSON work queue file.")
    parser.add_argument("--run-dir", type=Path, help="Private (0700) output directory for STATE.md, events.jsonl, heartbeat, PROOF.md.")
    parser.add_argument("--sync", choices=("local", "s3"), default="local", help="Artifact sync backend (default: local).")
    parser.add_argument("--floor-cu", type=float, default=500.0, help="Sprint-to-floor CU threshold (default: 500.0).")
    parser.add_argument("--steady-daily-cap-cu", type=float, default=65.0, help="Steady-state daily CU cap once <= floor-cu (default: 65.0).")
    parser.add_argument("--hard-stop-cu", type=float, default=50.0, help="Hard emergency CU floor (default: 50.0).")
    parser.add_argument("--max-total-slots", type=int, default=7, help="Max total simultaneous Colab assignments (default: 7).")
    parser.add_argument("--max-gpu-slots", type=int, default=3, help="Max concurrent GPU slots (default: 3).")
    parser.add_argument("--max-tpu-slots", type=int, default=2, help="Max concurrent TPU slots (default: 2).")
    parser.add_argument("--max-cpu-slots", type=int, default=4, help="Max concurrent CPU slots (default: 4).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned fleet dispatch without provisioning VMs.")
    parser.add_argument("--self-test", action="store_true", help="Run offline self-test with mocked Colab & S3 (0 CUs).")
    args = parser.parse_args(argv)

    if args.self_test:
        res = run_self_test()
        print(json.dumps(res, indent=2))
        return 0 if res.get("all_passed") else 1

    if not args.queue:
        parser.print_help()
        return 0

    raw_q = json.loads(args.queue.read_text(encoding="utf-8"))
    jobs = [FleetJob.from_dict(j, args.queue.parent) for j in raw_q.get("jobs", raw_q if isinstance(raw_q, list) else [])]
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="freetpu_fleet_"))
    gov_cfg = CuGovernorConfig(
        floor_cu=args.floor_cu,
        steady_daily_cap_cu=args.steady_daily_cap_cu,
        hard_stop_cu=args.hard_stop_cu,
        max_total_slots=args.max_total_slots,
        max_gpu_slots=args.max_gpu_slots,
        max_tpu_slots=args.max_tpu_slots,
        max_cpu_slots=args.max_cpu_slots,
    )
    disp = ColabFleetDispatcher(
        colab_bin=args.colab_bin,
        auth=args.auth,
        run_dir=run_dir,
        governor_cfg=gov_cfg,
        sync_mode=args.sync,
    )
    if args.dry_run:
        print(json.dumps(disp.plan_dry_run(jobs), indent=2))
        return 0

    proof = disp.run_queue(jobs)
    print(json.dumps(proof, indent=2))
    return 0 if proof["teardown"]["zero_fleet_leaked"] else 1


if __name__ == "__main__":
    sys.exit(main())
