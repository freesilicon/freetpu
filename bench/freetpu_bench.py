#!/usr/bin/env python3
"""
freetpu-bench: Unified cross-TPU microbenchmark suite and scorecard generator.

Inspired by NVIDIA's nvbench, nvbandwidth, nccl-tests, and dgx-spark-playbooks,
adapted for Pixel Tensor G5 (buenos_a0), Cloud TPU (v5e-1 / v6e-1), and GPU
comparison scorecards.

Suites:
  - compute   (nvbench-style): BF16 / INT8 / W8A16 GEMM, fused GEMM+SiLU,
              and small-op dispatch floor sweep.
  - bandwidth (nvbandwidth-style): HBM2e / LPDDR triad streaming bandwidth (GB/s)
              and host<->device transfer latency (us).
  - comms     (nccl-tests-style): Activation-hop latency (us) and throughput
              (GB/s) across 16 KB..4 MB PP/MoE payloads over TCP/USB-Ethernet.

Backends:
  - cpu         : Local NumPy FP32/INT8 reference execution.
  - buenos_mock : Local Tensor G5 (buenos_a0) mock running FP32 reference GEMM/SiLU
                  with 4x4 serpentine tile count metadata, host INT8 FP32-RNE
                  requantization, and FP32 W8A16 dequant.
  - jax_tpu     : Direct JAX/XLA execution on Cloud TPU (or JAX device).
  - colab_tpu   : Headless Google Colab TPU runner via `colab` CLI with isolated
                  `--config` state and automatic v6e1 -> v5e1 capacity fallback.
"""

# Note: Do not add `from __future__` imports to this file; `colab run` prepends
# executable prelude statements before compiling the script as one kernel cell.

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

SCHEMA_VERSION = "freetpu.bench.v1"
BENCH_JSON_BEGIN = "=== FREETPU_BENCH_JSON_BEGIN ==="
BENCH_JSON_END = "=== FREETPU_BENCH_JSON_END ==="

VALID_SUITES = ("compute", "bandwidth", "comms")
VALID_BACKENDS = ("cpu", "buenos_mock", "jax_tpu", "colab_tpu")

_DEFINITIVE_ALLOCATION_REFUSAL_MARKERS = (
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
)

_AMBIGUOUS_ALLOCATION_MARKERS = (
    "timed out",
    "timeout",
    "readtimeout",
    "connecttimeout",
    "retrying",
    "retry ",
    "retries",
    "http 502",
    "http 503",
    "http 504",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "connection reset",
    "connection aborted",
    "remote disconnected",
)


@dataclass(frozen=True)
class ReferenceBaseline:
    target_id: str
    display_name: str
    status: str  # Always "published-reference" for reference table rows
    memory_spec: str
    memory_gb: float
    peak_bf16_tflops: Optional[float]
    peak_int8_tops: Optional[float]
    measured_bf16_tflops: Optional[float]
    measured_int8_tops: Optional[float]
    bandwidth_gbps: float
    bandwidth_basis: str
    dispatch_floor_us: str
    hourly_cost_usd: float
    pricing_basis: str
    source_citation: str
    verified_decode_tok_s: Optional[float] = None
    verified_decode_label: Optional[str] = None


def _gib_to_decimal_gb(gib: float) -> float:
    """Convert a GiB capacity (2^30 bytes) to decimal GB (10^9 bytes) to match `--model-gb`."""
    return (gib * (1024**3)) / 1e9


PUBLISHED_BASELINES: Tuple[ReferenceBaseline, ...] = (
    ReferenceBaseline(
        target_id="pixel_tensor_g5",
        display_name="Pixel Tensor G5 (buenos_a0)",
        status="published-reference",
        memory_spec="15.6 GiB LPDDR (unified)",
        memory_gb=_gib_to_decimal_gb(15.6),
        peak_bf16_tflops=None,
        peak_int8_tops=None,
        measured_bf16_tflops=None,
        measured_int8_tops=0.993,
        bandwidth_gbps=27.96,
        bandwidth_basis="measured weight DMA (1x2048x2048 @ 150 us)",
        dispatch_floor_us="60.8-150 us hw (395 us wall)",
        hourly_cost_usd=0.05,
        pricing_basis="$1,299 Pixel 10 Pro amortized 3yr 24/7 ($0.049/hr) + $0.001/hr power (~5W); $0 cloud bill",
        source_citation="FreeTPU Phase 6 Report & buenos_a0 silicon probes (W8A16 BF16 1x64x32 = 411-426 us hw)",
        verified_decode_tok_s=5.85,
        verified_decode_label="Qwen2.5-1.5B-Q8_0 (tg64)",
    ),
    ReferenceBaseline(
        target_id="cloud_tpu_v5e_1",
        display_name="Cloud TPU v5e-1",
        status="published-reference",
        memory_spec="16 GiB HBM2e",
        memory_gb=_gib_to_decimal_gb(16.0),
        peak_bf16_tflops=197.0,
        peak_int8_tops=394.0,
        measured_bf16_tflops=179.36,
        measured_int8_tops=350.69,
        bandwidth_gbps=655.55,
        bandwidth_basis="measured triad BW (80.0% of 819 GB/s peak)",
        dispatch_floor_us="117.6-135.0 us sync (32 us pipelined)",
        hourly_cost_usd=1.20,
        pricing_basis="Google Cloud TPU v5e public on-demand (us-central1/us-east5/us-west1: $1.20/chip-hr)",
        source_citation="Google Cloud TPU v5e spec + FreeTPU Track B live v5e-1 silicon run",
        verified_decode_tok_s=97.53,
        verified_decode_label="Llama-3-8B W8A16 (B=1)",
    ),
    ReferenceBaseline(
        target_id="cloud_tpu_v6e_1",
        display_name="Cloud TPU v6e-1 (Trillium)",
        status="published-reference",
        memory_spec="31.25 GiB HBM2e (32 GiB)",
        memory_gb=_gib_to_decimal_gb(31.25),
        peak_bf16_tflops=918.0,
        peak_int8_tops=1836.0,
        measured_bf16_tflops=None,
        measured_int8_tops=None,
        bandwidth_gbps=1638.0,
        bandwidth_basis="published peak HBM bandwidth (1,638 GB/s)",
        dispatch_floor_us="~100 us PJRT est.",
        hourly_cost_usd=2.70,
        pricing_basis="Google Cloud TPU v6e public on-demand (us-east1/us-east5: $2.70/chip-hr)",
        source_citation="Google Cloud TPU v6e (Trillium) public architecture & pricing docs",
        verified_decode_tok_s=None,
        verified_decode_label=None,
    ),
    ReferenceBaseline(
        target_id="nvidia_h100_sxm",
        display_name="NVIDIA H100 SXM",
        status="published-reference",
        memory_spec="80 GiB HBM3",
        memory_gb=_gib_to_decimal_gb(80.0),
        peak_bf16_tflops=989.0,
        peak_int8_tops=1979.0,
        measured_bf16_tflops=None,
        measured_int8_tops=None,
        bandwidth_gbps=3350.0,
        bandwidth_basis="published peak HBM3 bandwidth (3,350 GB/s)",
        dispatch_floor_us="~5-15 us CUDA/Graph",
        hourly_cost_usd=3.49,
        pricing_basis="Specialized GPU cloud on-demand ($3.49/GPU-hr reference; hyperscaler 8-GPU VMs ~$7-11/GPU-hr)",
        source_citation="NVIDIA H100 SXM datasheet (dense Tensor Core, non-sparse) & public cloud rates",
        verified_decode_tok_s=None,
        verified_decode_label=None,
    ),
    ReferenceBaseline(
        target_id="nvidia_rtx_5090",
        display_name="NVIDIA RTX 5090",
        status="published-reference",
        memory_spec="32 GiB GDDR7",
        memory_gb=_gib_to_decimal_gb(32.0),
        peak_bf16_tflops=209.5,
        peak_int8_tops=838.0,
        measured_bf16_tflops=None,
        measured_int8_tops=None,
        bandwidth_gbps=1792.0,
        bandwidth_basis="published peak GDDR7 bandwidth (1,792 GB/s)",
        dispatch_floor_us="~5-15 us CUDA/Graph",
        hourly_cost_usd=0.50,
        pricing_basis="Specialized community GPU cloud ($0.50/GPU-hr midpoint of $0.32-$0.80/hr; $1,999 FE MSRP ~$0.20/hr self-hosted)",
        source_citation="NVIDIA RTX Blackwell architecture spec (dense non-sparse Tensor Core: 209.5 BF16 TF/s FP32-acc, 838 INT8 TOPS) & public community cloud pricing",
        verified_decode_tok_s=None,
        verified_decode_label=None,
    ),
)


def parse_suites(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --suite specification")
    if "all" in parts:
        return list(VALID_SUITES)
    out: List[str] = []
    for p in parts:
        if p not in VALID_SUITES:
            raise ValueError(f"Unknown suite {p!r}; expected one of {VALID_SUITES} or 'all'")
        if p not in out:
            out.append(p)
    return out


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("NumPy is required for freetpu_bench.py (`pip install numpy` or `apt install python3-numpy`).")
    return np


def _stats_summary(samples_us: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(x) for x in samples_us)
    if not ordered:
        return {"min_us": 0.0, "median_us": 0.0, "p95_us": 0.0, "max_us": 0.0}
    p95_idx = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered))) - 1))
    return {
        "min_us": round(ordered[0], 3),
        "median_us": round(statistics.median(ordered), 3),
        "p95_us": round(ordered[p95_idx], 3),
        "max_us": round(ordered[-1], 3),
    }


def _silu_np(x: Any) -> Any:
    # Numerically stable SiLU: x * sigmoid(x)
    x_f32 = x.astype(np.float32, copy=False)
    pos = x_f32 >= 0
    z = np.empty_like(x_f32)
    z[pos] = 1.0 / (1.0 + np.exp(-x_f32[pos]))
    exp_x = np.exp(x_f32[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return x_f32 * z


def _buenos_tile_count(m: int, k: int, n: int) -> int:
    """Compute Tensor G5 4x4 serpentine tile dispatches for an (M, K, N) GEMM."""
    # Tensor G5 buenos_a0 uses 4x4 PE grid (16 PEs); host runtime pads M to 1 or 64, N to 32.
    m_tiles = max(1, math.ceil(m / 64))
    k_tiles = max(1, math.ceil(k / 512))
    n_tiles = max(1, math.ceil(n / 128))
    return m_tiles * k_tiles * n_tiles


# ---------------------------------------------------------------------------
# Local CPU / buenos_mock suites
# ---------------------------------------------------------------------------


def run_compute_local(backend: str, quick: bool = False) -> Dict[str, Any]:
    _require_numpy()
    rng = np.random.default_rng(42)

    gemm_shapes = (
        [(1, 64, 32), (16, 128, 128), (64, 512, 512)]
        if quick
        else [(1, 64, 32), (16, 128, 128), (1, 576, 1536), (64, 512, 512), (256, 1024, 1024), (512, 1024, 1024)]
    )
    warmup = 2 if quick else 3
    iters = 4 if quick else 8

    cases: List[Dict[str, Any]] = []
    for m, k, n in gemm_shapes:
        flops = 2.0 * m * k * n

        # 1. FP32 reference GEMM
        a_f = rng.standard_normal((m, k), dtype=np.float32)
        b_f = rng.standard_normal((k, n), dtype=np.float32)
        for _ in range(warmup):
            _ = a_f @ b_f
        samples_us = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            _ = a_f @ b_f
            samples_us.append((time.perf_counter_ns() - t0) / 1e3)
        st = _stats_summary(samples_us)
        med_s = max(st["median_us"] * 1e-6, 1e-9)
        min_s = max(st["min_us"] * 1e-6, 1e-9)
        cases.append({
            "name": "gemm_fp32_ref" if backend == "cpu" else "gemm_fp32_mock",
            "shape": [m, k, n],
            "dtype": "fp32",
            "unit": "TFLOP/s",
            "median_perf": round((flops / med_s) / 1e12, 6),
            "peak_perf": round((flops / min_s) / 1e12, 6),
            "tiles": _buenos_tile_count(m, k, n) if backend == "buenos_mock" else None,
            **st,
        })

        # 2. INT32-widened integer GEMM reference from INT8 operands (+ FP32-RNE requant on buenos_mock)
        a_i8 = rng.integers(-64, 64, size=(m, k), dtype=np.int8)
        b_i8 = rng.integers(-64, 64, size=(k, n), dtype=np.int8)
        scale = np.float32(1.0 / max(1, k))
        for _ in range(warmup):
            acc = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
            if backend == "buenos_mock":
                _ = np.clip(np.rint(acc.astype(np.float32) * scale), -128, 127).astype(np.int8)
        samples_us = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            acc = a_i8.astype(np.int32) @ b_i8.astype(np.int32)
            if backend == "buenos_mock":
                _ = np.clip(np.rint(acc.astype(np.float32) * scale), -128, 127).astype(np.int8)
            samples_us.append((time.perf_counter_ns() - t0) / 1e3)
        st = _stats_summary(samples_us)
        med_s = max(st["median_us"] * 1e-6, 1e-9)
        min_s = max(st["min_us"] * 1e-6, 1e-9)
        int_basis = (
            "INT32-widened ref from INT8 + FP32-RNE requant (includes operand widening overhead)"
            if backend == "buenos_mock"
            else "INT32-widened ref from INT8 (includes operand widening overhead)"
        )
        cases.append({
            "name": "gemm_int32_ref_from_int8" if backend == "cpu" else "gemm_int32_mock_from_int8",
            "shape": [m, k, n],
            "dtype": "int8->int32_ref" if backend == "cpu" else "int8->int32_rne_mock",
            "precision_basis": int_basis,
            "unit": "TOPS",
            "median_perf": round((flops / med_s) / 1e12, 6),
            "peak_perf": round((flops / min_s) / 1e12, 6),
            "tiles": _buenos_tile_count(m, k, n) if backend == "buenos_mock" else None,
            **st,
        })

    # 3. W8A16 (u8 + zp=128 -> f32 + per-channel scale) + Fused FP32-ref GEMM+SiLU
    m, k, n = gemm_shapes[-1]
    flops = 2.0 * m * k * n
    a_w8 = rng.standard_normal((m, k), dtype=np.float32)
    b_f_last = rng.standard_normal((k, n), dtype=np.float32)
    w_u8 = rng.integers(0, 255, size=(k, n), dtype=np.uint8)
    w_scale = rng.uniform(0.005, 0.02, size=(1, n)).astype(np.float32)
    for _ in range(warmup):
        w_deq = (w_u8.astype(np.float32) - np.float32(128.0)) * w_scale
        _ = a_w8 @ w_deq
        _ = _silu_np(a_w8 @ b_f_last)
    samples_w8 = []
    samples_fused = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        w_deq = (w_u8.astype(np.float32) - np.float32(128.0)) * w_scale
        _ = a_w8 @ w_deq
        samples_w8.append((time.perf_counter_ns() - t0) / 1e3)

        t1 = time.perf_counter_ns()
        _ = _silu_np(a_w8 @ b_f_last)
        samples_fused.append((time.perf_counter_ns() - t1) / 1e3)

    st_w8 = _stats_summary(samples_w8)
    st_f = _stats_summary(samples_fused)
    cases.append({
        "name": "gemm_w8a16_dequant_fp32_ref" if backend == "cpu" else "gemm_w8a16_dequant_fp32_mock",
        "shape": [m, k, n],
        "dtype": "w8a16_fp32_ref",
        "unit": "TFLOP/s",
        "median_perf": round((flops / max(st_w8["median_us"] * 1e-6, 1e-9)) / 1e12, 6),
        "peak_perf": round((flops / max(st_w8["min_us"] * 1e-6, 1e-9)) / 1e12, 6),
        "tiles": _buenos_tile_count(m, k, n) if backend == "buenos_mock" else None,
        **st_w8,
    })
    cases.append({
        "name": "fused_gemm_silu_fp32_ref" if backend == "cpu" else "fused_gemm_silu_fp32_mock",
        "shape": [m, k, n],
        "dtype": "fp32+silu",
        "unit": "TFLOP/s",
        "median_perf": round((flops / max(st_f["median_us"] * 1e-6, 1e-9)) / 1e12, 6),
        "peak_perf": round((flops / max(st_f["min_us"] * 1e-6, 1e-9)) / 1e12, 6),
        "tiles": _buenos_tile_count(m, k, n) if backend == "buenos_mock" else None,
        **st_f,
    })

    # Dispatch floor is the median latency of the smallest 1x64x32 op.
    floor_cases = [c for c in cases if c["shape"] == [1, 64, 32] and c["name"].startswith("gemm_fp32")]
    peak_fp32 = max((c["peak_perf"] for c in cases if c["unit"] == "TFLOP/s"), default=0.0)
    peak_int32_ref = max((c["peak_perf"] for c in cases if c["unit"] == "TOPS"), default=0.0)
    int_basis = (
        "INT32-widened ref from INT8 + FP32-RNE requant (includes operand widening overhead)"
        if backend == "buenos_mock"
        else "INT32-widened ref from INT8 (includes operand widening overhead)"
    )
    return {
        "status": "measured",
        "backend": backend,
        "peak_bf16_tflops": None,
        "peak_fp32_tflops": round(peak_fp32, 4),
        "peak_int8_tops": None,
        "peak_int32_ref_tops": round(peak_int32_ref, 4),
        "int_precision_basis": int_basis,
        "dispatch_floor_us": floor_cases[0]["median_us"] if floor_cases else None,
        "cases": cases,
    }


def run_bandwidth_local(backend: str, quick: bool = False) -> Dict[str, Any]:
    _require_numpy()
    sizes_mb = [2, 8] if quick else [4, 16, 64]
    iters = 3 if quick else 6

    triad_cases: List[Dict[str, Any]] = []
    scalar = np.float32(1.618034)
    for mb in sizes_mb:
        count = (mb * 1024 * 1024) // 4
        a = np.ones(count, dtype=np.float32)
        b = np.full(count, 2.0, dtype=np.float32)
        c = np.empty(count, dtype=np.float32)
        # Warmup
        np.multiply(b, scalar, out=c)
        np.add(a, c, out=c)

        samples_us = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            np.multiply(b, scalar, out=c)
            np.add(a, c, out=c)
            samples_us.append((time.perf_counter_ns() - t0) / 1e3)

        tensor_bytes = count * 4
        effective_triad_bytes = tensor_bytes * 3  # Logical STREAM triad (2 reads + 1 write)
        actual_traffic_bytes = tensor_bytes * 5   # Two-pass NumPy: mul(b,s->c) [1R+1W] + add(a,c->c) [2R+1W]
        st = _stats_summary(samples_us)
        med_s = max(st["median_us"] * 1e-6, 1e-9)
        min_s = max(st["min_us"] * 1e-6, 1e-9)
        med_eff_gbps = (effective_triad_bytes / med_s) / 1e9
        peak_eff_gbps = (effective_triad_bytes / min_s) / 1e9
        med_act_gbps = (actual_traffic_bytes / med_s) / 1e9
        peak_act_gbps = (actual_traffic_bytes / min_s) / 1e9
        triad_cases.append({
            "name": "triad_fp32_2pass",
            "tensor_bytes": tensor_bytes,
            "effective_triad_bytes": effective_triad_bytes,
            "actual_traffic_bytes": actual_traffic_bytes,
            "traffic_model": "two_pass_numpy_5x_actual_3x_effective",
            "median_effective_gbps": round(med_eff_gbps, 3),
            "peak_effective_gbps": round(peak_eff_gbps, 3),
            "median_actual_gbps": round(med_act_gbps, 3),
            "peak_actual_gbps": round(peak_act_gbps, 3),
            "median_gbps": round(med_eff_gbps, 3),
            "peak_gbps": round(peak_eff_gbps, 3),
            "bandwidth_basis": "effective 3x STREAM triad (2-pass NumPy executes 5x array traffic)",
            **st,
        })

    # Host<->Device transfer simulation (aligned copy / staging overhead)
    transfer_sizes = [4096, 65536] if quick else [4096, 65536, 4 * 1024 * 1024]
    xfer_cases: List[Dict[str, Any]] = []
    for nbytes in transfer_sizes:
        src = np.arange(nbytes // 4, dtype=np.float32)
        dst = np.empty_like(src)
        np.copyto(dst, src)
        samples_us = []
        for _ in range(iters * 2):
            t0 = time.perf_counter_ns()
            np.copyto(dst, src)
            samples_us.append((time.perf_counter_ns() - t0) / 1e3)
        st = _stats_summary(samples_us)
        med_gbps = (nbytes / max(st["median_us"] * 1e-6, 1e-9)) / 1e9
        xfer_cases.append({
            "name": "host_stage_copy",
            "bytes": nbytes,
            "median_gbps": round(med_gbps, 3),
            **st,
        })

    peak_eff_bw = max((c["peak_effective_gbps"] for c in triad_cases), default=0.0)
    peak_act_bw = max((c["peak_actual_gbps"] for c in triad_cases), default=0.0)
    h2d_floor = xfer_cases[0]["median_us"] if xfer_cases else None
    return {
        "status": "measured",
        "backend": backend,
        "bandwidth_basis": "effective 3x STREAM triad (local 2-pass NumPy executes 5x array traffic)",
        "peak_triad_gbps": round(peak_eff_bw, 3),
        "peak_effective_triad_gbps": round(peak_eff_bw, 3),
        "peak_actual_traffic_gbps": round(peak_act_bw, 3),
        "host_device_floor_us": h2d_floor,
        "triad_cases": triad_cases,
        "transfer_cases": xfer_cases,
    }


# ---------------------------------------------------------------------------
# Comms suite (nccl-tests style activation-hop benchmark)
# ---------------------------------------------------------------------------


MAX_COMMS_PAYLOAD_BYTES = 4 * 1024 * 1024
COMMS_READ_TIMEOUT_S = 15.0
MAX_COMMS_CONCURRENT_CONNS = 8


def _recv_exact(sock: socket.socket, nbytes: int, deadline: Optional[float] = None) -> bytes:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    got = 0
    while got < nbytes:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise socket.timeout("Frame receive deadline exceeded")
            sock.settimeout(remaining)
        n = sock.recv_into(view[got:])
        if n == 0:
            raise EOFError("Socket closed while reading activation payload")
        got += n
    return bytes(buf)


def _handle_comms_conn(
    conn: socket.socket,
    sem: Optional[threading.BoundedSemaphore] = None,
    max_payload_bytes: int = MAX_COMMS_PAYLOAD_BYTES,
    read_timeout_s: float = COMMS_READ_TIMEOUT_S,
) -> None:
    try:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        while True:
            conn.settimeout(read_timeout_s)
            first = conn.recv(1)
            if not first:
                break
            frame_deadline = time.monotonic() + read_timeout_s
            hdr = first + _recv_exact(conn, 7, deadline=frame_deadline)
            (size,) = struct.unpack("!Q", hdr)
            if size == 0 or size > max_payload_bytes:
                break
            payload = _recv_exact(conn, size, deadline=frame_deadline)
            remaining = frame_deadline - time.monotonic()
            if remaining <= 0.0:
                raise socket.timeout("Frame send deadline exceeded")
            conn.settimeout(remaining)
            conn.sendall(hdr + payload)
    except (OSError, EOFError, socket.timeout):
        pass
    finally:
        try:
            conn.close()
        finally:
            if sem is not None:
                sem.release()


def run_comms_server(bind_host: str, port: int) -> None:
    """Run a persistent TCP activation-hop echo server for two-node benchmarks."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, port))
    srv.listen(8)
    actual_port = srv.getsockname()[1]
    sem = threading.BoundedSemaphore(MAX_COMMS_CONCURRENT_CONNS)
    print(f"freetpu-bench comms server listening on {bind_host}:{actual_port}", flush=True)
    try:
        while True:
            conn, _addr = srv.accept()
            if not sem.acquire(blocking=False):
                conn.close()
                continue
            try:
                t = threading.Thread(target=_handle_comms_conn, args=(conn, sem), daemon=True)
                t.start()
            except Exception:
                conn.close()
                sem.release()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


def run_comms_suite(
    host: str = "127.0.0.1",
    port: int = 0,
    quick: bool = False,
) -> Dict[str, Any]:
    """Measure PP/MoE activation-hop RTT, half-RTT ping-pong estimate, and throughput."""
    payload_sizes = (
        [16 * 1024, 64 * 1024]
        if quick
        else [16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
    )
    iters = 8 if quick else 20

    srv: Optional[socket.socket] = None
    srv_thread: Optional[threading.Thread] = None
    transport = "tcp_remote"

    if port == 0:
        transport = "tcp_loopback"
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, 0))
        srv.listen(4)
        port = srv.getsockname()[1]

        def _accept_one() -> None:
            assert srv is not None
            conn, _ = srv.accept()
            _handle_comms_conn(conn)

        srv_thread = threading.Thread(target=_accept_one, daemon=True)
        srv_thread.start()

    cases: List[Dict[str, Any]] = []
    try:
        with socket.create_connection((host, port), timeout=15.0) as cli:
            cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for size in payload_sizes:
                msg = struct.pack("!Q", size) + (b"\x5a" * size)
                expected_len = len(msg)
                # Warmup
                for _ in range(2):
                    cli.sendall(msg)
                    _ = _recv_exact(cli, expected_len)

                rtt_us: List[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter_ns()
                    cli.sendall(msg)
                    _ = _recv_exact(cli, expected_len)
                    rtt_us.append((time.perf_counter_ns() - t0) / 1e3)

                st_rtt = _stats_summary(rtt_us)
                # Ping-pong echo includes server turnaround and both directions, so RTT/2
                # is a half-RTT estimate rather than a measured directional one-way latency.
                half_rtt_med_us = st_rtt["median_us"] / 2.0
                # Ping-pong transfers `2 * size` bytes per RTT.
                bidir_gbps = ((2.0 * size) / max(st_rtt["median_us"] * 1e-6, 1e-9)) / 1e9
                peak_bidir_gbps = ((2.0 * size) / max(st_rtt["min_us"] * 1e-6, 1e-9)) / 1e9
                cases.append({
                    "payload_bytes": size,
                    "payload_label": f"{size // 1024} KB" if size < 1024 * 1024 else f"{size // (1024 * 1024)} MB",
                    "half_rtt_median_us": round(half_rtt_med_us, 3),
                    "rtt_min_us": st_rtt["min_us"],
                    "rtt_median_us": st_rtt["median_us"],
                    "rtt_p95_us": st_rtt["p95_us"],
                    "median_gbps": round(bidir_gbps, 3),
                    "peak_gbps": round(peak_bidir_gbps, 3),
                })
            # Clean shutdown signal
            cli.sendall(struct.pack("!Q", 0))
    finally:
        if srv_thread is not None:
            srv_thread.join(timeout=2.0)
        if srv is not None:
            srv.close()

    small_half_rtt = cases[0]["half_rtt_median_us"] if cases else None
    peak_gbps = max((c["peak_gbps"] for c in cases), default=0.0)
    return {
        "status": "measured",
        "host_platform": f"{platform.system()}-{platform.machine()}",
        "transport": transport,
        "target": f"{host}:{port}",
        "half_rtt_floor_us": small_half_rtt,
        "latency_basis": (
            "half-RTT ping-pong estimate (RTT/2; includes server turnaround and both directions, "
            "not measured directional one-way latency)"
        ),
        "peak_gbps": round(peak_gbps, 3),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# JAX / Cloud TPU suite (runs inside Colab or Cloud TPU VM)
# ---------------------------------------------------------------------------


def _safe_hbm_bytes_limit(dev: Any) -> Optional[int]:
    try:
        raw_ms = dev.memory_stats()
    except Exception:
        return None
    if isinstance(raw_ms, dict):
        val = raw_ms.get("bytes_limit")
        return val if isinstance(val, int) else None
    return None


def run_jax_tpu_suites(suites: Sequence[str], quick: bool = False, allow_jax_cpu: bool = False) -> Dict[str, Any]:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    devices = jax.devices()
    backend_name = jax.default_backend()
    if backend_name != "tpu" and not allow_jax_cpu:
        raise RuntimeError(
            f"JAX default backend is {backend_name!r}, not 'tpu'. "
            "Use `--backend colab_tpu` for remote Colab TPU or pass `--allow-jax-cpu`."
        )

    dev = devices[0]
    dev_kind = getattr(dev, "device_kind", backend_name)

    results: Dict[str, Any] = {
        "device_info": {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "default_backend": backend_name,
            "device_kind": dev_kind,
            "device_count": len(devices),
            "hbm_bytes_limit": _safe_hbm_bytes_limit(dev),
        }
    }

    if "compute" in suites:
        gemm_shapes = (
            [(1, 64, 32), (1024, 1024, 1024), (4096, 4096, 4096)]
            if quick
            else [
                (1, 64, 32),
                (16, 128, 128),
                (1, 576, 1536),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192),
            ]
        )
        warmup = 3 if quick else 5
        iters = 10 if quick else 25
        key = jax.random.PRNGKey(0)
        cases: List[Dict[str, Any]] = []

        @jax.jit
        def bf16_mm(x: Any, y: Any) -> Any:
            return jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16
            )

        @jax.jit
        def fused_mm_silu(x: Any, y: Any) -> Any:
            z = jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16
            )
            return z * jax.nn.sigmoid(z)

        @jax.jit
        def int8_mm(x: Any, y: Any) -> Any:
            return jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32
            )

        @jax.jit
        def w8a16_mm(x: Any, w_i8: Any, scale: Any) -> Any:
            w = w_i8.astype(jnp.bfloat16) * scale
            return jax.lax.dot_general(
                x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16
            )

        for m, k, n in gemm_shapes:
            flops = 2.0 * m * k * n
            a = jax.random.normal(key, (m, k), dtype=jnp.bfloat16)
            b = jax.random.normal(key, (k, n), dtype=jnp.bfloat16)
            a.block_until_ready()
            b.block_until_ready()
            for _ in range(warmup):
                bf16_mm(a, b).block_until_ready()
            samples_us = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                bf16_mm(a, b).block_until_ready()
                samples_us.append((time.perf_counter_ns() - t0) / 1e3)
            st = _stats_summary(samples_us)
            cases.append({
                "name": "gemm_bf16",
                "shape": [m, k, n],
                "dtype": "bf16",
                "unit": "TFLOP/s",
                "median_perf": round((flops / max(st["median_us"] * 1e-6, 1e-9)) / 1e12, 3),
                "peak_perf": round((flops / max(st["min_us"] * 1e-6, 1e-9)) / 1e12, 3),
                **st,
            })
            del a, b

        # INT8 + W8A16 + Fused GEMM+SiLU on major shapes
        major_shapes = [(4096, 4096, 4096)] if quick else [(4096, 4096, 4096), (8192, 8192, 8192)]
        for m, k, n in major_shapes:
            flops = 2.0 * m * k * n
            a_i8 = jax.random.randint(key, (m, k), -127, 127).astype(jnp.int8)
            b_i8 = jax.random.randint(key, (k, n), -127, 127).astype(jnp.int8)
            a_i8.block_until_ready()
            b_i8.block_until_ready()
            for _ in range(warmup):
                int8_mm(a_i8, b_i8).block_until_ready()
            samples_us = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                int8_mm(a_i8, b_i8).block_until_ready()
                samples_us.append((time.perf_counter_ns() - t0) / 1e3)
            st = _stats_summary(samples_us)
            cases.append({
                "name": "gemm_int8",
                "shape": [m, k, n],
                "dtype": "int8",
                "unit": "TOPS",
                "median_perf": round((flops / max(st["median_us"] * 1e-6, 1e-9)) / 1e12, 3),
                "peak_perf": round((flops / max(st["min_us"] * 1e-6, 1e-9)) / 1e12, 3),
                **st,
            })
            del a_i8, b_i8

        # Fused GEMM + SiLU & W8A16 on 4096^3
        m, k, n = 4096, 4096, 4096
        flops = 2.0 * m * k * n
        a_bf = jax.random.normal(key, (m, k), dtype=jnp.bfloat16)
        b_bf = jax.random.normal(key, (k, n), dtype=jnp.bfloat16)
        w_i8 = jax.random.randint(key, (k, n), -127, 127).astype(jnp.int8)
        scale = jnp.ones((1, n), dtype=jnp.bfloat16) * jnp.bfloat16(0.01)
        a_bf.block_until_ready()
        b_bf.block_until_ready()
        w_i8.block_until_ready()
        scale.block_until_ready()
        for _ in range(warmup):
            fused_mm_silu(a_bf, b_bf).block_until_ready()
            w8a16_mm(a_bf, w_i8, scale).block_until_ready()
        samples_fused, samples_w8 = [], []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fused_mm_silu(a_bf, b_bf).block_until_ready()
            samples_fused.append((time.perf_counter_ns() - t0) / 1e3)
            t1 = time.perf_counter_ns()
            w8a16_mm(a_bf, w_i8, scale).block_until_ready()
            samples_w8.append((time.perf_counter_ns() - t1) / 1e3)
        st_f = _stats_summary(samples_fused)
        st_w = _stats_summary(samples_w8)
        cases.append({
            "name": "fused_gemm_silu",
            "shape": [m, k, n],
            "dtype": "bf16+silu",
            "unit": "TFLOP/s",
            "median_perf": round((flops / max(st_f["median_us"] * 1e-6, 1e-9)) / 1e12, 3),
            "peak_perf": round((flops / max(st_f["min_us"] * 1e-6, 1e-9)) / 1e12, 3),
            **st_f,
        })
        cases.append({
            "name": "gemm_w8a16",
            "shape": [m, k, n],
            "dtype": "w8a16_bf16",
            "unit": "TFLOP/s",
            "median_perf": round((flops / max(st_w["median_us"] * 1e-6, 1e-9)) / 1e12, 3),
            "peak_perf": round((flops / max(st_w["min_us"] * 1e-6, 1e-9)) / 1e12, 3),
            **st_w,
        })

        floor_cases = [c for c in cases if c["shape"] == [1, 64, 32]]
        results["compute"] = {
            "status": "measured",
            "backend": "jax_tpu",
            "peak_bf16_tflops": round(max((c["peak_perf"] for c in cases if c["unit"] == "TFLOP/s"), default=0.0), 3),
            "peak_int8_tops": round(max((c["peak_perf"] for c in cases if c["unit"] == "TOPS"), default=0.0), 3),
            "dispatch_floor_us": floor_cases[0]["median_us"] if floor_cases else None,
            "cases": cases,
        }

    if "bandwidth" in suites:
        _require_numpy()
        key = jax.random.PRNGKey(1)
        # 64 MB, 256 MB, 1024 MB BF16 tensors (in elements of 2 bytes)
        sizes_mb = [64, 256] if quick else [64, 256, 1024]
        warmup = 3 if quick else 5
        iters = 10 if quick else 20

        @jax.jit
        def triad(x: Any, y: Any, s: Any) -> Any:
            return x + s * y

        s_val = jnp.bfloat16(1.618)
        triad_cases: List[Dict[str, Any]] = []
        for mb in sizes_mb:
            elems = (mb * 1024 * 1024) // 2
            a = jax.random.normal(key, (elems,), dtype=jnp.bfloat16)
            b = jax.random.normal(key, (elems,), dtype=jnp.bfloat16)
            a.block_until_ready()
            b.block_until_ready()
            for _ in range(warmup):
                triad(a, b, s_val).block_until_ready()
            samples_us = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                triad(a, b, s_val).block_until_ready()
                samples_us.append((time.perf_counter_ns() - t0) / 1e3)
            st = _stats_summary(samples_us)
            total_bytes = elems * 2 * 3  # 2 reads + 1 write
            triad_cases.append({
                "name": "triad_bf16",
                "tensor_bytes": elems * 2,
                "total_traffic_bytes": total_bytes,
                "median_gbps": round((total_bytes / max(st["median_us"] * 1e-6, 1e-9)) / 1e9, 3),
                "peak_gbps": round((total_bytes / max(st["min_us"] * 1e-6, 1e-9)) / 1e9, 3),
                **st,
            })
            del a, b

        # Host <-> Device transfer overhead
        xfer_cases: List[Dict[str, Any]] = []
        for nbytes in ([4096, 4 * 1024 * 1024] if quick else [4096, 65536, 4 * 1024 * 1024]):
            host_arr = np.ones(nbytes // 2, dtype=np.float16)
            h2d_samples, d2h_samples = [], []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                d_arr = jax.device_put(host_arr, dev)
                d_arr.block_until_ready()
                h2d_samples.append((time.perf_counter_ns() - t0) / 1e3)
                t1 = time.perf_counter_ns()
                _ = np.asarray(d_arr)
                d2h_samples.append((time.perf_counter_ns() - t1) / 1e3)
            st_h2d = _stats_summary(h2d_samples)
            st_d2h = _stats_summary(d2h_samples)
            xfer_cases.append({
                "name": "h2d_device_put",
                "bytes": nbytes,
                "median_gbps": round((nbytes / max(st_h2d["median_us"] * 1e-6, 1e-9)) / 1e9, 3),
                **st_h2d,
            })
            xfer_cases.append({
                "name": "d2h_to_host",
                "bytes": nbytes,
                "median_gbps": round((nbytes / max(st_d2h["median_us"] * 1e-6, 1e-9)) / 1e9, 3),
                **st_d2h,
            })

        results["bandwidth"] = {
            "status": "measured",
            "backend": "jax_tpu",
            "bandwidth_basis": "fused 3x STREAM triad (2R+1W on TPU HBM)",
            "peak_triad_gbps": round(max((c["peak_gbps"] for c in triad_cases), default=0.0), 3),
            "host_device_floor_us": xfer_cases[0]["median_us"] if xfer_cases else None,
            "triad_cases": triad_cases,
            "transfer_cases": xfer_cases,
        }

    return results


# ---------------------------------------------------------------------------
# Headless Colab TPU runner (--backend colab_tpu)
# ---------------------------------------------------------------------------


def resolve_colab_bin(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    found = shutil.which("colab")
    if found:
        return found
    home_bin = Path.home() / ".local" / "bin" / "colab"
    if home_bin.is_file():
        return str(home_bin)
    return "colab"


def extract_bench_json(stdout: str) -> Optional[Dict[str, Any]]:
    pattern = re.compile(
        re.escape(BENCH_JSON_BEGIN) + r"\s*(.*?)\s*" + re.escape(BENCH_JSON_END),
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


def is_colab_allocation_error(returncode: int, stderr: str) -> bool:
    """Return True only for definitive allocation refusals where no runtime was assigned.

    Ambiguous failures (timeouts, HTTP 502/503/504, connection resets, or any retried
    allocation attempt) return False because the server may have created a billable
    runtime before the error or subsequent refusal.
    """
    if returncode == 0:
        return False
    low = stderr.lower()
    if "[colab] session ready" in low:
        return False
    if any(marker in low for marker in _AMBIGUOUS_ALLOCATION_MARKERS):
        return False
    return any(marker in low for marker in _DEFINITIVE_ALLOCATION_REFUSAL_MARKERS)


def build_colab_run_commands(
    colab_bin: str,
    auth: str,
    config_path: Path,
    tpu_order: Sequence[str],
    timeout_s: float,
    remote_suites: Sequence[str],
    quick: bool,
    session_base: str,
    script_path: Path,
) -> List[Tuple[str, str, List[str]]]:
    cmds: List[Tuple[str, str, List[str]]] = []
    suite_arg = ",".join(remote_suites)
    for idx, tpu in enumerate(tpu_order):
        sess = f"{session_base}-{tpu}" if idx > 0 else session_base
        cmd = [
            colab_bin,
            "--auth",
            auth,
            "--config",
            str(config_path),
            "run",
            "-s",
            sess,
            "--tpu",
            tpu.upper(),
            "--keep",
            "--timeout",
            str(timeout_s),
            str(script_path),
            "--suite",
            suite_arg,
            "--backend",
            "jax_tpu",
            "--emit-markers",
        ]
        if quick:
            cmd.append("--quick")
        cmds.append((tpu, sess, cmd))
    return cmds


_COLAB_SESSION_LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s+(\S+)\s*\|")


def _parse_colab_sessions(sessions_stdout: str) -> List[Tuple[str, str]]:
    """Parse `colab sessions` lines formatted as `[name] endpoint | Hardware: ...`."""
    entries: List[Tuple[str, str]] = []
    for line in sessions_stdout.splitlines():
        m = _COLAB_SESSION_LINE_RE.match(line)
        if m:
            entries.append((m.group(1).strip(), m.group(2).strip()))
    return entries


def _token_present(text: str, token: str) -> bool:
    pat = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])")
    return any(pat.search(line) for line in text.splitlines())


def _session_still_listed(sessions_stdout: str, session_name: str) -> bool:
    return _token_present(sessions_stdout, session_name)


MIN_COLAB_CLI_VERSION: Tuple[int, int, int] = (0, 7, 6)
_COLAB_SEMVER_RE = re.compile(
    r"(?i)\b(\d+)\.(\d+)\.(\d+)(?P<pre>(?:[-.]?(?:a|b|rc|dev|alpha|beta)[0-9]*))?"
)


def parse_colab_cli_version(output: str) -> Optional[Tuple[Tuple[int, int, int], bool, str]]:
    """Parse a semantic version string from `colab version` output.

    Returns `((major, minor, patch), is_prerelease, matched_text)` or `None`.
    """
    m = _COLAB_SEMVER_RE.search(output)
    if not m:
        return None
    ver = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    is_pre = bool(m.group("pre"))
    return ver, is_pre, m.group(0)


def _check_colab_cli_version(
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    colab_bin: str,
) -> str:
    """Enforce `google-colab-cli >= 0.7.6` before making any authenticated call.

    Releases <= 0.7.5 log authorization headers and runtime-proxy tokens to
    `~/.config/colab-cli/colab.log` under the default umask. Fail closed when
    the version is below 0.7.6 or cannot be determined.
    """
    ver_res = runner([colab_bin, "version"])
    combined = f"{ver_res.stdout}\n{ver_res.stderr}".strip()
    if ver_res.returncode != 0:
        raise RuntimeError(
            f"Refusing to run authenticated Colab commands: failed to verify google-colab-cli version "
            f"via `{colab_bin} version` (rc={ver_res.returncode}): {combined[-300:]}. "
            f"Install google-colab-cli >= 0.7.6 (`uv tool install 'google-colab-cli>=0.7.6'`)."
        )
    parsed = parse_colab_cli_version(ver_res.stdout) or parse_colab_cli_version(combined)
    if parsed is None:
        raise RuntimeError(
            f"Refusing to run authenticated Colab commands: could not parse google-colab-cli version "
            f"from `{colab_bin} version` output ({combined!r}). "
            f"Install google-colab-cli >= 0.7.6 (`uv tool install 'google-colab-cli>=0.7.6'`)."
        )
    ver_tuple, is_pre, ver_text = parsed
    if ver_tuple < MIN_COLAB_CLI_VERSION or (ver_tuple == MIN_COLAB_CLI_VERSION and is_pre):
        raise RuntimeError(
            f"Refusing to run authenticated Colab commands: google-colab-cli >= 0.7.6 is required "
            f"(detected {ver_text}; releases <= 0.7.5 log authorization headers and runtime-proxy tokens "
            f"under the default umask). Upgrade via `uv tool install 'google-colab-cli>=0.7.6'` or "
            f"`pip install -U 'google-colab-cli>=0.7.6'`."
        )
    return ver_text


def _format_colab_recovery_hint(
    colab_bin: str,
    auth: str,
    iso_cfg: Path,
    session_name: str,
    leaked_endpoints: Sequence[str],
    orphaned_endpoints: Sequence[str],
    unattributed_endpoints: Sequence[str] = (),
) -> str:
    sessions_cmd = shlex.join([colab_bin, "--auth", auth, "--config", str(iso_cfg), "sessions"])
    stop_cmd = shlex.join([colab_bin, "--auth", auth, "--config", str(iso_cfg), "stop", "-s", session_name])
    mapped_leaked = [ep for ep in leaked_endpoints if ep not in orphaned_endpoints]
    parts: List[str] = []
    if orphaned_endpoints:
        ep_list = ", ".join(orphaned_endpoints)
        parts.append(
            f"Confirmed owned orphaned `[?]` endpoint(s) [{ep_list}] have no local session mapping "
            f"(so `colab stop -s {session_name}` will no-op); terminate them in Colab "
            f"(https://colab.research.google.com -> Runtime -> Manage sessions) or via Colab's "
            f"unassign endpoint (`POST /tun/m/unassign/<endpoint>`), then confirm removal with `{sessions_cmd}`."
        )
    if unattributed_endpoints:
        unattr_list = ", ".join(unattributed_endpoints)
        parts.append(
            f"Unattributed `[?]` endpoint(s) [{unattr_list}] appeared without an established session mapping "
            f"(could belong to this run or another concurrent workload on the same account); "
            f"inspect `{sessions_cmd}` and Colab Runtime -> Manage sessions to confirm ownership before terminating any runtime."
        )
    if mapped_leaked:
        parts.append(
            f"Session config preserved at {iso_cfg}; retry `{stop_cmd}` "
            f"(or inspect Colab Runtime -> Manage sessions) and verify with `{sessions_cmd}`."
        )
    elif not orphaned_endpoints and not unattributed_endpoints:
        parts.append(
            f"Allocation/teardown outcome could not be confirmed (a runtime may still be provisioning); "
            f"session config preserved at {iso_cfg}. Re-check `{sessions_cmd}` (and Colab Runtime -> "
            f"Manage sessions) after provisioning settles, and retry `{stop_cmd}` if `{session_name}` appears."
        )
    return " ".join(parts)


class _ColabSigtermInterrupt(BaseException):
    """Raised when SIGTERM arrives during a retained Colab TPU run."""


COLAB_PROVISIONING_GRACE_S = 180.0
COLAB_CONTROL_TIMEOUT_S = 90.0
COLAB_UNBOUNDED_FALLBACK_DEADLINE_S = 1800.0


def _terminate_proc_group(proc: subprocess.Popen[str]) -> Tuple[str, str]:
    """Terminate and reap a child process (and its session group on POSIX)."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        pass
    try:
        return proc.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        return proc.communicate()


def _default_colab_runner(
    cmd: Sequence[str],
    timeout_s: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Colab CLI command non-interactively with an optional host-side deadline."""
    env = os.environ.copy()
    env.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "60")
    # Unconditionally disable internal Colab CLI allocation retries so a timed-out
    # POST cannot be retried and masked by a subsequent HTTP 412 refusal while the
    # first runtime is still provisioning server-side.
    env["COLAB_CLI_ASSIGN_RETRIES"] = "0"
    popen_kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(list(cmd), **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        out, err = _terminate_proc_group(proc)
        raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=float(timeout_s or 0.0), output=out, stderr=err)
    except BaseException:
        _terminate_proc_group(proc)
        raise
    return subprocess.CompletedProcess(list(cmd), proc.returncode, stdout, stderr)


def _make_default_colab_runner(
    run_timeout_s: float,
    provisioning_grace_s: float = COLAB_PROVISIONING_GRACE_S,
    control_timeout_s: float = COLAB_CONTROL_TIMEOUT_S,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    """Build a non-interactive Colab runner with a host-side wall-clock deadline."""
    base_run = run_timeout_s if run_timeout_s > 0.0 else COLAB_UNBOUNDED_FALLBACK_DEADLINE_S
    run_deadline = base_run + max(0.0, provisioning_grace_s)

    def _runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        limit = run_deadline if "run" in cmd else control_timeout_s
        return _default_colab_runner(cmd, timeout_s=limit)

    return _runner


def _verify_and_stop_session(
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    colab_bin: str,
    auth: str,
    iso_cfg: Path,
    session_name: str,
    pre_endpoints: Optional[Set[str]] = None,
    session_was_ready: bool = False,
    definitive_refusal: bool = False,
) -> Tuple[bool, Optional[str], List[str], List[str], List[str]]:
    """Stop a Colab runtime and verify its backend endpoint disappeared.

    Returns `(teardown_verified, observed_endpoint, leaked_endpoints,
    orphaned_endpoints, unattributed_endpoints)`.
    Because `colab run` is invoked with `--keep`, the local session-to-endpoint
    mapping stays intact in `iso_cfg` until we query `colab sessions`, capture
    the backend runtime endpoint, invoke `colab stop -s`, and verify that the
    endpoint no longer appears under either `[session_name]` or `[?]`.
    Only endpoints mapped to `session_name` are classified as confirmed leaks;
    newly appearing `[?]` endpoints without established ownership are recorded
    as `unattributed_endpoints` so unrelated concurrent runtimes are never
    recommended for blind termination.
    """
    def _run_teardown_cmd(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return runner(cmd)
        except subprocess.TimeoutExpired as exc:
            out_s = exc.output.decode("utf-8", errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
            err_s = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return subprocess.CompletedProcess(
                list(cmd),
                124,
                out_s,
                f"timed out after {exc.timeout}s: {err_s}".strip(),
            )

    before = _run_teardown_cmd([colab_bin, "--auth", auth, "--config", str(iso_cfg), "sessions"])
    before_ok = before.returncode == 0
    before_entries = _parse_colab_sessions(before.stdout) if before_ok else []
    tracked_endpoints: Set[str] = {ep for name, ep in before_entries if name == session_name}
    if before_ok and pre_endpoints is not None:
        new_unattributed: Set[str] = {
            ep for name, ep in before_entries if name == "?" and ep not in pre_endpoints
        }
    elif before_ok and session_was_ready and not tracked_endpoints:
        new_unattributed = {ep for name, ep in before_entries if name == "?"}
    else:
        new_unattributed = set()

    observed_endpoint = sorted(tracked_endpoints)[0] if tracked_endpoints else None

    needs_stop = (
        (not before_ok)
        or bool(tracked_endpoints)
        or _session_still_listed(before.stdout, session_name)
        or session_was_ready
    )
    if not needs_stop:
        if not definitive_refusal:
            # Ambiguous failure (e.g. ReadTimeout, HTTP 502/503/504, connection reset, or
            # non-refusal error before `[colab] Session READY`): even when the first `colab
            # sessions` check shows zero new endpoints, server-side provisioning may still be
            # in progress and appear moments later. Keep teardown unverified, preserve state,
            # block fallback, and surface recovery guidance to re-check after provisioning settles.
            return False, observed_endpoint, [], [], sorted(new_unattributed)
        # Definitive allocation refusal (e.g. HTTP 412) did not create a runtime for this run;
        # keep any concurrently started `[?]` assignments unattributed so fallback is not blocked.
        return True, observed_endpoint, [], [], sorted(new_unattributed)

    stop_res = _run_teardown_cmd([colab_bin, "--auth", auth, "--config", str(iso_cfg), "stop", "-s", session_name])
    recheck = _run_teardown_cmd([colab_bin, "--auth", auth, "--config", str(iso_cfg), "sessions"])
    if recheck.returncode != 0:
        return False, observed_endpoint, sorted(tracked_endpoints), [], sorted(new_unattributed)

    recheck_entries = _parse_colab_sessions(recheck.stdout)
    recheck_endpoints: Set[str] = {ep for _, ep in recheck_entries}
    recheck_orphans: Set[str] = {ep for name, ep in recheck_entries if name == "?"}

    leaked: Set[str] = set()
    orphaned: Set[str] = set()
    unattributed: Set[str] = set()

    if tracked_endpoints:
        # This run's endpoint ownership is confirmed: verify that endpoint specifically.
        for ep in tracked_endpoints:
            if ep in recheck_endpoints or _token_present(recheck.stdout, ep):
                leaked.add(ep)
                if ep in recheck_orphans:
                    orphaned.add(ep)
        for name, ep in recheck_entries:
            if name == session_name:
                leaked.add(ep)
            elif name == "?" and pre_endpoints is not None and ep not in pre_endpoints and ep not in tracked_endpoints:
                unattributed.add(ep)
    else:
        # No named endpoint was captured: keep `[?]` assignments unattributed (unconfirmed ownership),
        # while marking teardown unverified because this run reached ready/interrupted state.
        for ep in new_unattributed:
            if ep in recheck_endpoints or _token_present(recheck.stdout, ep):
                unattributed.add(ep)
        for name, ep in recheck_entries:
            if name == session_name:
                leaked.add(ep)
            elif name == "?" and pre_endpoints is not None and ep not in pre_endpoints:
                unattributed.add(ep)
            elif name == "?" and pre_endpoints is None and session_was_ready:
                unattributed.add(ep)

    name_still_listed = _session_still_listed(recheck.stdout, session_name)
    if leaked or name_still_listed:
        return False, observed_endpoint, sorted(leaked), sorted(orphaned), sorted(unattributed)

    if not tracked_endpoints and unattributed:
        return False, observed_endpoint, [], [], sorted(unattributed)

    if not tracked_endpoints and not definitive_refusal:
        if not before_ok and stop_res.returncode != 0:
            return False, observed_endpoint, [], [], sorted(unattributed)
        if before_ok and not _session_still_listed(before.stdout, session_name):
            # Neither `before` nor a local session name confirmed ownership of a stopped endpoint
            # (e.g. interrupted while provisioning was still in flight). Keep teardown unverified.
            return False, observed_endpoint, [], [], sorted(unattributed)

    if stop_res.returncode != 0 and not tracked_endpoints and pre_endpoints is None and recheck_entries:
        return False, observed_endpoint, [], [], sorted(unattributed)

    return True, observed_endpoint, [], [], sorted(unattributed)


def run_colab_tpu_suites(
    suites: Sequence[str],
    quick: bool,
    colab_bin: str,
    auth: str,
    tpu_order: Sequence[str],
    config_path: Optional[Path],
    timeout_s: float,
    dry_run: bool = False,
    runner: Optional[Callable[[Sequence[str]], subprocess.CompletedProcess[str]]] = None,
) -> Dict[str, Any]:
    remote_suites = [s for s in suites if s in ("compute", "bandwidth")]
    if not remote_suites:
        if dry_run:
            return {
                "colab_meta": {
                    "status": "dry_run",
                    "tpu_order": list(tpu_order),
                    "planned_commands": [],
                    "note": "No remote Colab suites requested (comms runs locally).",
                }
            }
        return {}

    if runner is None:
        runner = _make_default_colab_runner(timeout_s)

    tmp_dir: Optional[Path] = None
    if config_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="freetpu-bench-colab-"))
        iso_cfg = tmp_dir / "colab-sessions.json"
    else:
        iso_cfg = Path(os.path.abspath(os.path.expanduser(str(config_path))))
        iso_cfg.parent.mkdir(parents=True, exist_ok=True)
    preserve_tmp_dir = False
    session_base = f"freetpu-bench-{uuid.uuid4().hex[:8]}"
    script_path = Path(__file__).resolve()

    planned = build_colab_run_commands(
        colab_bin=colab_bin,
        auth=auth,
        config_path=iso_cfg,
        tpu_order=tpu_order,
        timeout_s=timeout_s,
        remote_suites=remote_suites,
        quick=quick,
        session_base=session_base,
        script_path=script_path,
    )

    if dry_run:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return {
            "colab_meta": {
                "status": "dry_run",
                "tpu_order": list(tpu_order),
                "planned_commands": [cmd for _, _, cmd in planned],
            }
        }

    try:
        colab_cli_version = _check_colab_cli_version(runner, colab_bin)
        usage_res = runner([colab_bin, "--auth", auth, "--config", str(iso_cfg), "usage"])
        if usage_res.returncode != 0:
            detail = (usage_res.stderr or usage_res.stdout).strip()[-600:]
            signin_hint = (
                f"Run `{shlex.join([colab_bin, '--auth', 'oauth2', 'usage'])}` interactively first to complete OAuth sign-in."
                if auth == "oauth2"
                else "Ensure ADC credentials are valid (e.g. `gcloud auth application-default login` with Colab scopes) or switch to `--colab-auth oauth2` after interactive sign-in."
            )
            raise RuntimeError(f"colab usage preflight failed (non-interactive): {detail}. {signin_hint}")

        attempts: List[Dict[str, Any]] = []
        for idx, (tpu, sess, cmd) in enumerate(planned):
            pre_res = runner([colab_bin, "--auth", auth, "--config", str(iso_cfg), "sessions"])
            if pre_res.returncode != 0:
                detail = (pre_res.stderr or pre_res.stdout).strip()[-600:]
                raise RuntimeError(
                    f"colab sessions preflight failed (rc={pre_res.returncode}) before allocating {tpu}: {detail}"
                )
            pre_endpoints: Set[str] = {ep for _, ep in _parse_colab_sessions(pre_res.stdout)}
            res: Optional[subprocess.CompletedProcess[str]] = None
            interrupted: Optional[BaseException] = None
            teardown_verified = False
            endpoint: Optional[str] = None
            leaked_endpoints: List[str] = []
            orphaned_endpoints: List[str] = []
            unattributed_endpoints: List[str] = []
            recovery_hint: Optional[str] = None

            install_sigterm = (
                hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread()
            )
            prev_sigterm: Any = None
            sigterm_state = {"in_teardown": False, "deferred": False}

            def _sigterm_handler(_signum: int, _frame: Any) -> None:
                if sigterm_state["in_teardown"]:
                    sigterm_state["deferred"] = True
                    return
                raise _ColabSigtermInterrupt("Received SIGTERM during Colab TPU run")

            if install_sigterm:
                prev_sigterm = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, _sigterm_handler)
            try:
                try:
                    res = runner(cmd)
                except BaseException as exc:
                    interrupted = exc
                finally:
                    sigterm_state["in_teardown"] = True
                    session_was_ready = (
                        (res is not None and (res.returncode == 0 or "[colab] session ready" in res.stderr.lower()))
                        or (interrupted is not None)
                    )
                    definitive_refusal = (
                        res is not None
                        and interrupted is None
                        and is_colab_allocation_error(res.returncode, res.stderr)
                    )
                    try:
                        (
                            teardown_verified,
                            endpoint,
                            leaked_endpoints,
                            orphaned_endpoints,
                            unattributed_endpoints,
                        ) = _verify_and_stop_session(
                            runner,
                            colab_bin,
                            auth,
                            iso_cfg,
                            sess,
                            pre_endpoints=pre_endpoints,
                            session_was_ready=session_was_ready,
                            definitive_refusal=definitive_refusal,
                        )
                    except BaseException:
                        preserve_tmp_dir = True
                        recovery_hint = _format_colab_recovery_hint(
                            colab_bin, auth, iso_cfg, sess, leaked_endpoints, orphaned_endpoints, unattributed_endpoints
                        )
                        print(
                            f"[freetpu-bench] WARNING: Colab teardown interrupted for session={sess}. {recovery_hint}",
                            file=sys.stderr,
                        )
                        raise
                    if not teardown_verified:
                        preserve_tmp_dir = True
                        recovery_hint = _format_colab_recovery_hint(
                            colab_bin, auth, iso_cfg, sess, leaked_endpoints, orphaned_endpoints, unattributed_endpoints
                        )
                        print(
                            f"[freetpu-bench] WARNING: Colab teardown unverified for session={sess} "
                            f"endpoint={endpoint} leaked_endpoints={leaked_endpoints} "
                            f"orphaned_endpoints={orphaned_endpoints} "
                            f"unattributed_endpoints={unattributed_endpoints}. {recovery_hint}",
                            file=sys.stderr,
                        )
                    elif unattributed_endpoints:
                        print(
                            f"[freetpu-bench] INFO: Verified teardown of session={sess} endpoint={endpoint}; "
                            f"observed unattributed concurrent session(s)={unattributed_endpoints} (not terminated).",
                            file=sys.stderr,
                        )
            finally:
                if install_sigterm:
                    signal.signal(signal.SIGTERM, prev_sigterm)

            if interrupted is not None:
                if isinstance(interrupted, subprocess.TimeoutExpired):
                    raise RuntimeError(
                        f"Colab TPU run exceeded host-side deadline ({interrupted.timeout:.1f}s) on {tpu} "
                        f"(teardown_verified={teardown_verified}, endpoint={endpoint}, "
                        f"leaked_endpoints={leaked_endpoints}, unattributed_endpoints={unattributed_endpoints})"
                    ) from interrupted
                raise interrupted
            if sigterm_state["deferred"]:
                raise _ColabSigtermInterrupt("Received SIGTERM during Colab teardown")
            assert res is not None

            attempt_rec: Dict[str, Any] = {
                "tpu": tpu,
                "session": sess,
                "endpoint": endpoint,
                "returncode": res.returncode,
                "teardown_verified": teardown_verified,
                "leaked_endpoints": leaked_endpoints,
                "orphaned_endpoints": orphaned_endpoints,
                "unattributed_endpoints": unattributed_endpoints,
                "stderr_tail": res.stderr.strip()[-600:],
            }
            if not teardown_verified:
                attempt_rec["recovery_config"] = str(iso_cfg)
                attempt_rec["recovery_hint"] = recovery_hint
            attempts.append(attempt_rec)

            if res.returncode == 0:
                payload = extract_bench_json(res.stdout)
                if payload is None:
                    raise RuntimeError(
                        f"Colab session {sess} exited 0 without {BENCH_JSON_BEGIN} payload."
                    )
                meta: Dict[str, Any] = {
                    "status": "ok",
                    "colab_cli_version": colab_cli_version,
                    "tpu": tpu,
                    "session": sess,
                    "endpoint": endpoint,
                    "teardown_verified": teardown_verified,
                    "leaked_endpoints": leaked_endpoints,
                    "orphaned_endpoints": orphaned_endpoints,
                    "unattributed_endpoints": unattributed_endpoints,
                    "usage_preflight": usage_res.stdout.strip(),
                    "attempts": attempts,
                }
                if not teardown_verified:
                    meta["recovery_config"] = str(iso_cfg)
                    meta["recovery_hint"] = recovery_hint
                payload["colab_meta"] = meta
                return payload
            if idx + 1 < len(planned) and teardown_verified and is_colab_allocation_error(res.returncode, res.stderr):
                continue
            raise RuntimeError(
                f"Colab TPU run failed on {tpu} (rc={res.returncode}, teardown_verified={teardown_verified}, "
                f"leaked_endpoints={leaked_endpoints}, orphaned_endpoints={orphaned_endpoints}): "
                f"{res.stderr.strip()[-600:]}"
            )
        raise RuntimeError(f"All Colab TPU candidates exhausted: {attempts}")
    finally:
        if tmp_dir is not None and not preserve_tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Economics & Scorecard Builder
# ---------------------------------------------------------------------------


def validate_economics_params(model_gb: float, bw_efficiency: float, batch_size: int) -> None:
    if not math.isfinite(model_gb) or model_gb <= 0.0:
        raise ValueError(f"--model-gb must be finite and > 0 (got {model_gb!r})")
    if not math.isfinite(bw_efficiency) or not (0.0 < bw_efficiency <= 1.0):
        raise ValueError(f"--bw-efficiency must be finite and in (0, 1] (got {bw_efficiency!r})")
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1 (got {batch_size!r})")


def _compute_economics(
    bf16_tflops: Optional[float],
    int8_tops: Optional[float],
    bw_gbps: Optional[float],
    hourly_cost_usd: Optional[float],
    model_gb: float,
    bw_efficiency: float,
    batch_size: int,
    verified_tok_s: Optional[float] = None,
    memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    validate_economics_params(model_gb, bw_efficiency, batch_size)
    weights_exceed_memory = (memory_gb is not None) and (model_gb > memory_gb)
    single_device_fits: Optional[bool] = (not weights_exceed_memory) if memory_gb is not None else None
    projection_note = (
        f"unavailable: {model_gb:g} GB weights exceed {memory_gb:.2f} GB single-device memory (requires sharding/offload)"
        if weights_exceed_memory and memory_gb is not None
        else None
    )
    if hourly_cost_usd is None or hourly_cost_usd <= 0.0:
        return {
            "tflops_per_dollar_hr": None,
            "int8_tops_per_dollar_hr": None,
            "gbps_per_dollar_hr": None,
            "est_decode_tok_s": None,
            "usd_per_1m_tokens_model": None,
            "usd_per_1m_tokens_verified": None,
            "single_device_fits": single_device_fits,
            "cost_projection_note": projection_note,
        }
    tf_per_dollar = round(bf16_tflops / hourly_cost_usd, 2) if bf16_tflops is not None else None
    int8_per_dollar = round(int8_tops / hourly_cost_usd, 2) if int8_tops is not None else None
    gbps_per_dollar = round(bw_gbps / hourly_cost_usd, 2) if bw_gbps is not None else None

    # Weight-bandwidth-limited single-device decode model:
    # Suppress single-device projections when active weights alone exceed target memory capacity.
    est_tok_s: Optional[float] = None
    usd_per_1m_model: Optional[float] = None
    if not weights_exceed_memory and bw_gbps is not None and bw_gbps > 0.0:
        est_tok_s = round(((bw_gbps * bw_efficiency) / model_gb) * max(1, batch_size), 2)
        if est_tok_s > 0.0:
            usd_per_1m_model = round((hourly_cost_usd / (est_tok_s * 3600.0)) * 1e6, 3)

    usd_per_1m_verified: Optional[float] = None
    if verified_tok_s is not None and verified_tok_s > 0.0:
        usd_per_1m_verified = round((hourly_cost_usd / (verified_tok_s * 3600.0)) * 1e6, 3)

    return {
        "tflops_per_dollar_hr": tf_per_dollar,
        "int8_tops_per_dollar_hr": int8_per_dollar,
        "gbps_per_dollar_hr": gbps_per_dollar,
        "est_decode_tok_s": est_tok_s,
        "usd_per_1m_tokens_model": usd_per_1m_model,
        "usd_per_1m_tokens_verified": usd_per_1m_verified,
        "single_device_fits": single_device_fits,
        "cost_projection_note": projection_note,
    }


def _infer_tpu_fallback_memory_gib(device_kind: str, colab_tpu: Optional[str] = None) -> Optional[float]:
    """Return documented usable HBM capacity in GiB when JAX `memory_stats()` is unavailable."""
    kind_low = " ".join(str(device_kind).lower().split())
    if colab_tpu == "v6e1" or "v6e" in kind_low or "v6 lite" in kind_low or "trillium" in kind_low:
        return 31.25
    if colab_tpu == "v5e1" or "v5e" in kind_low or "v5 lite" in kind_low:
        return 16.0
    if "v5p" in kind_low or re.search(r"\bv5\b", kind_low):
        return 95.0
    return None


def _infer_tpu_hourly_cost_usd(device_kind: str, colab_tpu: Optional[str] = None) -> Optional[float]:
    """Map JAX `device_kind` / Colab SKU to public GCP on-demand $/chip-hr."""
    kind_low = " ".join(str(device_kind).lower().split())
    if colab_tpu == "v6e1" or "v6e" in kind_low or "v6 lite" in kind_low or "trillium" in kind_low:
        return 2.70
    if colab_tpu == "v5e1" or "v5e" in kind_low or "v5 lite" in kind_low:
        return 1.20
    if "v5p" in kind_low or re.search(r"\bv5\b", kind_low):
        return 4.20
    return None


def build_scorecard_rows(
    report: Dict[str, Any],
    model_gb: float = 8.0,
    bw_efficiency: float = 0.70,
    batch_size: int = 1,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    comp = report.get("compute") or {}
    bw = report.get("bandwidth") or {}
    comms = report.get("comms") or {}
    dev_info = report.get("device_info") or {}
    backend = report.get("backend", "cpu")
    colab_tpu = (report.get("colab_meta") or {}).get("tpu")

    host_platform = report.get("host_platform") or f"{platform.system()}-{platform.machine()}"

    # 1. Live measured row(s)
    if backend in ("jax_tpu", "colab_tpu"):
        # Only emit a TPU row when actual TPU compute or bandwidth results exist.
        if comp or bw:
            kind = dev_info.get("device_kind") or (colab_tpu.upper() if colab_tpu else "Cloud TPU")
            label = f"LIVE RUN: {kind} ({backend})"
            inferred_cost = _infer_tpu_hourly_cost_usd(str(kind), colab_tpu)
            cost_basis = f"GCP public on-demand rate (${inferred_cost:.2f}/chip-hr)" if inferred_cost else "local/unspecified"
            mem_limit = dev_info.get("hbm_bytes_limit")
            fallback_gib = _infer_tpu_fallback_memory_gib(str(kind), colab_tpu)
            mem_str = (
                f"{mem_limit / (1024**3):.2f} GiB HBM"
                if mem_limit
                else (f"{fallback_gib:g} GiB HBM (spec)" if fallback_gib is not None else "TPU HBM")
            )
            # Compare against --model-gb in matching decimal GB (reserve GiB strictly for display).
            live_mem_gb_exact: Optional[float] = (
                (float(mem_limit) / 1e9)
                if mem_limit
                else (_gib_to_decimal_gb(fallback_gib) if fallback_gib is not None else None)
            )
            live_mem_gb: Optional[float] = round(live_mem_gb_exact, 2) if live_mem_gb_exact is not None else None

            bf16_tf = comp.get("peak_bf16_tflops")
            fp32_tf = comp.get("peak_fp32_tflops")
            int8_tops = comp.get("peak_int8_tops")
            bw_gbps = bw.get("peak_triad_gbps")
            floor_us = comp.get("dispatch_floor_us")

            econ = _compute_economics(
                bf16_tflops=bf16_tf,
                int8_tops=int8_tops,
                bw_gbps=bw_gbps,
                hourly_cost_usd=inferred_cost,
                model_gb=model_gb,
                bw_efficiency=bw_efficiency,
                batch_size=batch_size,
                memory_gb=live_mem_gb_exact,
            )
            tpu_row: Dict[str, Any] = {
                "target": label,
                "row_type": "measured",
                "memory": mem_str,
                "memory_gb": live_mem_gb,
                "bf16_tflops": bf16_tf,
                "fp32_tflops": fp32_tf,
                "int8_tops": int8_tops,
                "bw_gbps": bw_gbps,
                "dispatch_floor_us": f"{floor_us} us" if floor_us is not None else "—",
                # Local TCP comms never runs on remote TPU ICI; attribute separately below.
                "half_rtt_est_us": "—",
                "hourly_cost_usd": inferred_cost,
                "pricing_basis": cost_basis,
                **econ,
            }
            if bw.get("bandwidth_basis"):
                tpu_row["bw_basis"] = bw["bandwidth_basis"]
            rows.append(tpu_row)

        if comms:
            comms_host = comms.get("host_platform") or host_platform
            comms_transport = comms.get("transport") or "tcp"
            half_rtt_us = comms.get("half_rtt_floor_us")
            econ = _compute_economics(
                bf16_tflops=None,
                int8_tops=None,
                bw_gbps=None,
                hourly_cost_usd=None,
                model_gb=model_gb,
                bw_efficiency=bw_efficiency,
                batch_size=batch_size,
            )
            rows.append({
                "target": f"LIVE RUN: Local Host Comms ({comms_host}, {comms_transport})",
                "row_type": "measured",
                "memory": "host RAM",
                "bf16_tflops": None,
                "fp32_tflops": None,
                "int8_tops": None,
                "bw_gbps": None,
                "dispatch_floor_us": "—",
                "half_rtt_est_us": f"{half_rtt_us} us (half-RTT est.)" if half_rtt_us is not None else "—",
                "comms_host_platform": comms_host,
                "comms_transport": comms_transport,
                "comms_target": comms.get("target"),
                "comms_peak_gbps": comms.get("peak_gbps"),
                "hourly_cost_usd": None,
                "pricing_basis": f"local host comms ({comms_transport})",
                **econ,
            })
    elif comp or bw or comms:
        if backend == "buenos_mock":
            label = f"LIVE RUN: Host CPU via buenos_mock ({platform.machine()})"
            inferred_cost = None
            cost_basis = "host mock (zero TPU silicon)"
            mem_str = "host RAM (mock)"
        else:
            label = f"LIVE RUN: Host CPU ({platform.machine()})"
            inferred_cost = None
            cost_basis = "local host CPU"
            mem_str = "host RAM"

        bf16_tf = comp.get("peak_bf16_tflops")
        fp32_tf = comp.get("peak_fp32_tflops")
        int8_tops = comp.get("peak_int8_tops")
        int32_ref_tops = comp.get("peak_int32_ref_tops")
        bw_gbps = bw.get("peak_triad_gbps")
        floor_us = comp.get("dispatch_floor_us")
        half_rtt_us = comms.get("half_rtt_floor_us")

        econ = _compute_economics(
            bf16_tflops=bf16_tf,
            int8_tops=int8_tops,
            bw_gbps=bw_gbps,
            hourly_cost_usd=inferred_cost,
            model_gb=model_gb,
            bw_efficiency=bw_efficiency,
            batch_size=batch_size,
        )
        host_row: Dict[str, Any] = {
            "target": label,
            "row_type": "measured",
            "memory": mem_str,
            "bf16_tflops": bf16_tf,
            "fp32_tflops": fp32_tf,
            "int8_tops": int8_tops,
            "int32_ref_tops": int32_ref_tops,
            "bw_gbps": bw_gbps,
            "dispatch_floor_us": f"{floor_us} us" if floor_us is not None else "—",
            "half_rtt_est_us": f"{half_rtt_us} us (half-RTT est.)" if half_rtt_us is not None else "—",
            "hourly_cost_usd": inferred_cost,
            "pricing_basis": cost_basis,
            **econ,
        }
        if comp.get("int_precision_basis"):
            host_row["int_precision_basis"] = comp["int_precision_basis"]
        if bw.get("bandwidth_basis"):
            host_row["bw_basis"] = bw["bandwidth_basis"]
        if comms:
            host_row["comms_transport"] = comms.get("transport")
        rows.append(host_row)

    # 2. Published reference rows
    for base in PUBLISHED_BASELINES:
        bf16_ref = base.measured_bf16_tflops if base.measured_bf16_tflops is not None else base.peak_bf16_tflops
        int8_ref = base.measured_int8_tops if base.measured_int8_tops is not None else base.peak_int8_tops
        econ = _compute_economics(
            bf16_tflops=bf16_ref,
            int8_tops=int8_ref,
            bw_gbps=base.bandwidth_gbps,
            hourly_cost_usd=base.hourly_cost_usd,
            model_gb=model_gb,
            bw_efficiency=bw_efficiency,
            batch_size=batch_size,
            verified_tok_s=base.verified_decode_tok_s,
            memory_gb=base.memory_gb,
        )
        rows.append({
            "target": base.display_name,
            "row_type": base.status,
            "memory": base.memory_spec,
            "memory_gb": round(base.memory_gb, 2),
            "bf16_tflops": bf16_ref,
            "bf16_is_measured": base.measured_bf16_tflops is not None,
            "int8_tops": int8_ref,
            "int8_is_measured": base.measured_int8_tops is not None,
            "bw_gbps": base.bandwidth_gbps,
            "bw_basis": base.bandwidth_basis,
            "dispatch_floor_us": base.dispatch_floor_us,
            "half_rtt_est_us": "60-512 us (local/USB-C)" if base.target_id == "pixel_tensor_g5" else "ICI / NVLink / PCIe",
            "hourly_cost_usd": base.hourly_cost_usd,
            "pricing_basis": base.pricing_basis,
            "source_citation": base.source_citation,
            "verified_decode_label": base.verified_decode_label,
            "verified_decode_tok_s": base.verified_decode_tok_s,
            **econ,
        })

    return rows


def _short_bw_basis_tag(row: Dict[str, Any]) -> str:
    basis = str(row.get("bw_basis") or "").lower()
    if row.get("row_type") == "measured":
        if "2-pass" in basis:
            return "3x eff 2-pass"
        return "measured triad"
    if "weight dma" in basis:
        return "measured DMA"
    if "triad" in basis and "measured" in basis:
        return "measured triad"
    if "measured" in basis:
        return "measured"
    return "peak-spec"


def render_markdown(report: Dict[str, Any]) -> str:
    econ_cfg = report.get("economics_assumptions") or {}
    model_gb = econ_cfg.get("model_gb", 8.0)
    bw_eff = econ_cfg.get("bw_efficiency", 0.70)
    batch_size = econ_cfg.get("batch_size", 1)
    rows = report.get("scorecard") or []

    colab_meta = report.get("colab_meta") or {}
    remote_host = colab_meta.get("remote_host_platform")
    host_str = (
        f"{report.get('host_platform')} (remote: {remote_host})"
        if remote_host
        else str(report.get("host_platform"))
    )

    lines: List[str] = [
        "# `freetpu-bench` Cross-Silicon Scorecard",
        "",
        f"- **Timestamp (UTC)**: `{report.get('timestamp_utc')}`",
        f"- **Backend**: `{report.get('backend')}` | **Host**: `{host_str}` | **Suites**: `{', '.join(report.get('suites', []))}`",
        "",
    ]

    if colab_meta.get("status") == "dry_run":
        planned_cmds = colab_meta.get("planned_commands") or []
        lines.append("## Planned Colab TPU Commands (Dry Run)")
        lines.append("")
        if planned_cmds:
            lines.append(f"- **TPU fallback order**: `{', '.join(colab_meta.get('tpu_order') or [])}`")
            lines.append("")
            lines.append("```bash")
            for cmd in planned_cmds:
                lines.append(shlex.join(cmd))
            lines.append("```")
        else:
            lines.append(str(colab_meta.get("note") or "No remote Colab commands planned."))
        lines.append("")

    lines.extend([
        "## 1. Cross-Hardware Scorecard (`measured` vs `published-reference`)",
        "",
        "| Target | Row Type | Memory | BF16 (TF/s) | INT8 (TOPS) | Mem BW (GB/s) | Dispatch Floor | $/hr | TF/s per $ | Est. $/1M tok |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])

    for r in rows:
        is_live = r.get("row_type") == "measured"
        if r.get("bf16_tflops") is not None:
            bf16_tag = "measured" if (is_live or r.get("bf16_is_measured")) else "peak-spec"
            bf16_s = f"{r['bf16_tflops']:.2f} ({bf16_tag})"
        elif r.get("fp32_tflops") is not None:
            bf16_s = f"{r['fp32_tflops']:.2f} (fp32)"
        else:
            bf16_s = "—"
        if r.get("int8_tops") is not None:
            int8_tag = "measured" if (is_live or r.get("int8_is_measured")) else "peak-spec"
            int8_s = f"{r['int8_tops']:.2f} ({int8_tag})"
        elif r.get("int32_ref_tops") is not None:
            int8_s = f"{r['int32_ref_tops']:.2f} (int32-widened ref)"
        else:
            int8_s = "—"
        if r.get("bw_gbps") is not None:
            bw_s = f"{r['bw_gbps']:.2f} ({_short_bw_basis_tag(r)})"
        else:
            bw_s = "—"
        cost_s = f"${r['hourly_cost_usd']:.2f}" if r.get("hourly_cost_usd") is not None else "—"
        tf_d_s = f"{r['tflops_per_dollar_hr']:.1f}" if r.get("tflops_per_dollar_hr") is not None else (
            f"{r['int8_tops_per_dollar_hr']:.1f} (I8)" if r.get("int8_tops_per_dollar_hr") is not None else "—"
        )
        tok_cost = r.get("usd_per_1m_tokens_model")
        if tok_cost is not None:
            tok_s = f"${tok_cost:.2f}"
        elif r.get("single_device_fits") is False:
            tok_s = "N/A (>mem)"
        else:
            tok_s = "—"
        lines.append(
            f"| {r['target']} | `{r['row_type']}` | {r['memory']} | {bf16_s} | {int8_s} | {bw_s} | {r['dispatch_floor_us']} | {cost_s} | {tf_d_s} | {tok_s} |"
        )

    lines.extend([
        "",
        f"> **Normalized `$/1M tok` Assumptions**: `{model_gb:.1f} GB` active weight read per token (e.g. Llama-3-8B W8A16), `{bw_eff * 100:.0f}%` memory-bandwidth decode efficiency, `batch_size={batch_size}` (`tok/s = BW * eff * B / model_gb`). Published references distinguish peak spec vs verified silicon runs. Single-device projections show `N/A (>mem)` when `model_gb` exceeds a target's known memory capacity (multi-device sharding or offload required). Local host `(int32-widened ref)` cells widen `int8 -> int32` operands inside the timed NumPy matmul (including conversion overhead) rather than executing native INT8 accelerator ops.",
        "",
    ])

    comp = report.get("compute")
    if comp:
        lines.extend([
            "## 2. Live `compute` Suite (`nvbench`-style)",
            "",
            "| Kernel | Shape `(M,K,N)` | Dtype | Median Lat (us) | Min Lat (us) | Median Perf | Peak Perf |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for c in comp.get("cases", []):
            shape_s = "x".join(str(x) for x in c["shape"])
            lines.append(
                f"| `{c['name']}` | `{shape_s}` | `{c['dtype']}` | {c['median_us']:.2f} | {c['min_us']:.2f} | {c['median_perf']} {c['unit']} | {c['peak_perf']} {c['unit']} |"
            )
        lines.append("")

    bw = report.get("bandwidth")
    if bw:
        lines.extend([
            "## 3. Live `bandwidth` Suite (`nvbandwidth`-style)",
            "",
            "| Probe | Payload / Traffic | Median Lat (us) | Median BW (GB/s) | Peak BW (GB/s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for c in bw.get("triad_cases", []):
            mb = c["tensor_bytes"] / (1024 * 1024)
            if "effective_triad_bytes" in c and "actual_traffic_bytes" in c:
                eff_mb = c["effective_triad_bytes"] / (1024 * 1024)
                act_mb = c["actual_traffic_bytes"] / (1024 * 1024)
                traffic_s = f"{mb:.1f} MiB tensor ({eff_mb:.1f} MiB 3x eff / {act_mb:.1f} MiB 5x actual 2-pass)"
                med_s = f"{c['median_effective_gbps']:.3f} eff ({c['median_actual_gbps']:.3f} act)"
                peak_s = f"{c['peak_effective_gbps']:.3f} eff ({c['peak_actual_gbps']:.3f} act)"
            else:
                traffic_s = f"{mb:.1f} MiB tensor ({c['total_traffic_bytes'] / (1024*1024):.1f} MiB 3x fused)"
                med_s = f"{c['median_gbps']:.3f}"
                peak_s = f"{c['peak_gbps']:.3f}"
            lines.append(
                f"| `{c['name']}` | {traffic_s} | {c['median_us']:.2f} | {med_s} | {peak_s} |"
            )
        for c in bw.get("transfer_cases", []):
            lines.append(
                f"| `{c['name']}` | {c['bytes']} B | {c['median_us']:.2f} | {c['median_gbps']:.3f} | — |"
            )
        lines.append("")

    comms = report.get("comms")
    if comms:
        lines.extend([
            f"## 4. Live `comms` Suite (`nccl-tests`-style, `{comms.get('transport')}`)",
            "",
            "| Payload | Half-RTT Est. Median (us) | RTT Median (us) | RTT p95 (us) | Median Bidir (GB/s) | Peak Bidir (GB/s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for c in comms.get("cases", []):
            lines.append(
                f"| `{c['payload_label']}` | {c['half_rtt_median_us']:.2f} | {c['rtt_median_us']:.2f} | {c['rtt_p95_us']:.2f} | {c['median_gbps']:.3f} | {c['peak_gbps']:.3f} |"
            )
        lines.extend([
            "",
            "> **Ping-Pong Half-RTT Note**: `Half-RTT Est.` is `RTT / 2` from the TCP send-and-echo probe (includes server turnaround and both directions; it is a ping-pong latency estimate, not a measured directional one-way latency on asymmetric links).",
            "",
        ])

    lines.extend([
        "## 5. Published Pricing & Hardware Citations",
        "",
    ])
    for b in PUBLISHED_BASELINES:
        lines.append(f"- **{b.display_name}**: {b.pricing_basis}; BW basis: {b.bandwidth_basis} — *{b.source_citation}*.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline Self-Test
# ---------------------------------------------------------------------------


def run_self_test() -> int:
    _require_numpy()
    # 1. Verify suite parsing & baseline metadata integrity
    assert parse_suites("all") == ["compute", "bandwidth", "comms"]
    assert parse_suites("compute,comms") == ["compute", "comms"]
    assert len(PUBLISHED_BASELINES) == 5
    for b in PUBLISHED_BASELINES:
        assert b.status == "published-reference"
        assert b.hourly_cost_usd > 0
        assert b.pricing_basis and b.source_citation

    # 2. Verify cpu + buenos_mock quick execution + comms loopback
    for backend in ("cpu", "buenos_mock"):
        comp = run_compute_local(backend, quick=True)
        assert comp["status"] == "measured"
        assert comp["peak_bf16_tflops"] is None
        assert comp["peak_fp32_tflops"] > 0
        assert comp["peak_int8_tops"] is None
        assert comp["peak_int32_ref_tops"] > 0
        assert "INT32-widened ref" in comp["int_precision_basis"]
        suffix = "ref" if backend == "cpu" else "mock"
        assert any(c["name"] == f"gemm_fp32_{suffix}" and c["dtype"] == "fp32" for c in comp["cases"])
        assert any(c["name"] == f"gemm_int32_{suffix}_from_int8" and "int32" in c["dtype"] for c in comp["cases"])
        assert any(c["name"] == f"fused_gemm_silu_fp32_{suffix}" for c in comp["cases"])
        assert any(c["name"] == f"gemm_w8a16_dequant_fp32_{suffix}" for c in comp["cases"])

        bw = run_bandwidth_local(backend, quick=True)
        assert bw["status"] == "measured"
        assert bw["peak_triad_gbps"] > 0
        assert bw["peak_effective_triad_gbps"] == bw["peak_triad_gbps"]
        assert bw["peak_actual_traffic_gbps"] > bw["peak_effective_triad_gbps"]
        for tc in bw["triad_cases"]:
            assert tc["name"] == "triad_fp32_2pass"
            assert tc["effective_triad_bytes"] == tc["tensor_bytes"] * 3
            assert tc["actual_traffic_bytes"] == tc["tensor_bytes"] * 5
            assert "total_traffic_bytes" not in tc
            assert tc["peak_actual_gbps"] > tc["peak_effective_gbps"]

    rtx5090 = next(b for b in PUBLISHED_BASELINES if b.target_id == "nvidia_rtx_5090")
    assert rtx5090.peak_bf16_tflops == 209.5
    assert rtx5090.peak_int8_tops == 838.0

    # Comms size-cap & timeout hardening
    c_sock, s_sock = socket.socketpair()
    t_srv = threading.Thread(
        target=_handle_comms_conn,
        args=(s_sock, None, 1024, 0.2),
        daemon=True,
    )
    t_srv.start()
    c_sock.sendall(struct.pack("!Q", 2048))
    assert c_sock.recv(16) == b""
    t_srv.join(timeout=2.0)
    assert not t_srv.is_alive()
    c_sock.close()

    c_to, s_to = socket.socketpair()
    t_to = threading.Thread(
        target=_handle_comms_conn,
        args=(s_to, None, 1024, 0.05),
        daemon=True,
    )
    t_to.start()
    c_to.sendall(struct.pack("!Q", 64) + b"ab")
    assert c_to.recv(16) == b""
    t_to.join(timeout=2.0)
    assert not t_to.is_alive()
    c_to.close()

    # Slow-drip partial TCP frame MUST hit overall frame deadline (cannot hold worker slot forever)
    c_drip, s_drip = socket.socketpair()
    t_drip = threading.Thread(
        target=_handle_comms_conn,
        args=(s_drip, None, 1024, 0.12),
        daemon=True,
    )
    t_drip.start()
    c_drip.sendall(struct.pack("!Q", 64))
    t_drip_start = time.monotonic()
    for _ in range(15):
        if not t_drip.is_alive():
            break
        try:
            c_drip.sendall(b"x")
        except OSError:
            break
        time.sleep(0.03)
    t_drip.join(timeout=1.5)
    assert not t_drip.is_alive()
    assert time.monotonic() - t_drip_start < 1.0
    c_drip.close()

    # _safe_hbm_bytes_limit + TPU kind pricing discrimination (v5p vs v5e vs v6e)
    class _NoneMemDev:
        def memory_stats(self) -> None:
            return None

    class _ErrMemDev:
        def memory_stats(self) -> Dict[str, int]:
            raise RuntimeError("no mem stats")

    assert _safe_hbm_bytes_limit(_NoneMemDev()) is None
    assert _safe_hbm_bytes_limit(_ErrMemDev()) is None

    for dev_kind, colab_kind, exp_cost in (
        ("TPU v5", None, 4.20),
        ("TPU v5p", None, 4.20),
        ("TPU v5 lite", None, 1.20),
        ("TPU v5 lite", "v5e1", 1.20),
        ("TPU v6 lite", None, 2.70),
        ("TPU v6e", "v6e1", 2.70),
        ("TPU v4", None, None),
    ):
        probe_rows = build_scorecard_rows({
            "backend": "jax_tpu",
            "device_info": {"device_kind": dev_kind, "hbm_bytes_limit": 16 * 1024**3},
            "colab_meta": {"tpu": colab_kind} if colab_kind else {},
            "compute": {"peak_bf16_tflops": 100.0},
        })
        assert probe_rows[0]["hourly_cost_usd"] == exp_cost, (dev_kind, probe_rows[0])

    comms = run_comms_suite(host="127.0.0.1", port=0, quick=True)
    assert comms["status"] == "measured"
    assert comms["half_rtt_floor_us"] is not None and comms["half_rtt_floor_us"] > 0
    assert "half-RTT" in comms["latency_basis"]
    assert len(comms["cases"]) == 2
    assert "half_rtt_median_us" in comms["cases"][0]
    assert "one_way_median_us" not in comms["cases"][0]

    # 3. Verify scorecard + Markdown/JSON rendering + clean-room output
    report = {
        "schema": SCHEMA_VERSION,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "backend": "buenos_mock",
        "host_platform": f"{platform.system()}-{platform.machine()}",
        "suites": ["compute", "bandwidth", "comms"],
        "economics_assumptions": {"model_gb": 8.0, "bw_efficiency": 0.70, "batch_size": 1},
        "compute": comp,
        "bandwidth": bw,
        "comms": comms,
    }
    report["scorecard"] = build_scorecard_rows(report)
    measured_row = next(r for r in report["scorecard"] if r["row_type"] == "measured")
    assert measured_row["bf16_tflops"] is None
    assert measured_row["fp32_tflops"] > 0
    assert measured_row["int8_tops"] is None
    assert measured_row["int32_ref_tops"] > 0
    assert "INT32-widened ref" in measured_row["int_precision_basis"]
    assert "bf16_is_fp32_ref" not in measured_row
    rtx5090_row = next(r for r in report["scorecard"] if r["target"] == "NVIDIA RTX 5090")
    assert rtx5090_row["int8_tops"] == 838.0
    assert rtx5090_row["int8_tops_per_dollar_hr"] == 1676.0
    assert sum(1 for r in report["scorecard"] if r["row_type"] == "published-reference") == 5
    md = render_markdown(report)
    assert "`measured`" in md and "`published-reference`" in md
    assert "(fp32)" in md and "(int32-widened ref)" in md
    assert "179.36 (measured)" in md and "655.55 (measured triad)" in md
    assert "989.00 (peak-spec)" in md and "3350.00 (peak-spec)" in md
    assert "27.96 (measured DMA)" in md and "3x eff 2-pass" in md
    assert "3x eff /" in md and "5x actual 2-pass" in md
    assert "Half-RTT Est. Median (us)" in md and "1-Way Hop Median" not in md
    assert "Pixel Tensor G5" in md and "Cloud TPU v6e-1" in md and "NVIDIA H100 SXM" in md

    # 3d. Verify single-device projections are suppressed when model_gb exceeds device memory capacity
    big_report = dict(report)
    big_report["economics_assumptions"] = {"model_gb": 40.0, "bw_efficiency": 0.70, "batch_size": 1}
    big_report["scorecard"] = build_scorecard_rows(big_report, model_gb=40.0)
    by_target = {r["target"]: r for r in big_report["scorecard"]}
    for small_target in (
        "Pixel Tensor G5 (buenos_a0)",
        "Cloud TPU v5e-1",
        "Cloud TPU v6e-1 (Trillium)",
        "NVIDIA RTX 5090",
    ):
        row = by_target[small_target]
        assert row["single_device_fits"] is False, (small_target, row)
        assert row["est_decode_tok_s"] is None, (small_target, row)
        assert row["usd_per_1m_tokens_model"] is None, (small_target, row)
        assert row["cost_projection_note"] and "exceed" in row["cost_projection_note"]
    h100_big = by_target["NVIDIA H100 SXM"]
    assert h100_big["single_device_fits"] is True
    assert h100_big["est_decode_tok_s"] is not None and h100_big["est_decode_tok_s"] > 0
    assert h100_big["usd_per_1m_tokens_model"] is not None and h100_big["usd_per_1m_tokens_model"] > 0
    big_md = render_markdown(big_report)
    assert "N/A (>mem)" in big_md

    # Live v6e (with and without memory_stats) and v6e reference MUST agree in decimal GB (~33.55 GB for 31.25 GiB)
    v6e_live_33 = build_scorecard_rows(
        {
            "backend": "jax_tpu",
            "device_info": {"device_kind": "TPU v6 lite", "hbm_bytes_limit": int(31.25 * 1024**3)},
            "compute": {"peak_bf16_tflops": 900.0},
            "bandwidth": {"peak_triad_gbps": 1400.0},
        },
        model_gb=33.0,
    )
    v6e_fallback_33 = build_scorecard_rows(
        {
            "backend": "colab_tpu",
            "colab_meta": {"tpu": "v6e1"},
            "device_info": {"device_kind": "TPU v6 lite", "hbm_bytes_limit": None},
            "compute": {"peak_bf16_tflops": 900.0},
            "bandwidth": {"peak_triad_gbps": 1400.0},
        },
        model_gb=33.0,
    )
    v6e_ref_33 = next(r for r in v6e_live_33 if r["target"] == "Cloud TPU v6e-1 (Trillium)")
    assert v6e_live_33[0]["memory"] == "31.25 GiB HBM"
    assert v6e_fallback_33[0]["memory"] == "31.25 GiB HBM (spec)"
    for v6_row in (v6e_live_33[0], v6e_fallback_33[0], v6e_ref_33):
        assert v6_row["memory_gb"] == 33.55, v6_row
        assert v6_row["single_device_fits"] is True, v6_row
        assert v6_row["est_decode_tok_s"] is not None and v6_row["est_decode_tok_s"] > 0, v6_row
        assert v6_row["usd_per_1m_tokens_model"] is not None and v6_row["usd_per_1m_tokens_model"] > 0, v6_row

    v6e_live_34 = build_scorecard_rows(
        {
            "backend": "jax_tpu",
            "device_info": {"device_kind": "TPU v6 lite", "hbm_bytes_limit": int(31.25 * 1024**3)},
            "compute": {"peak_bf16_tflops": 900.0},
            "bandwidth": {"peak_triad_gbps": 1400.0},
        },
        model_gb=34.0,
    )
    v6e_fallback_34 = build_scorecard_rows(
        {
            "backend": "colab_tpu",
            "colab_meta": {"tpu": "v6e1"},
            "device_info": {"device_kind": "TPU v6 lite", "hbm_bytes_limit": None},
            "compute": {"peak_bf16_tflops": 900.0},
            "bandwidth": {"peak_triad_gbps": 1400.0},
        },
        model_gb=34.0,
    )
    v6e_ref_34 = next(r for r in v6e_live_34 if r["target"] == "Cloud TPU v6e-1 (Trillium)")
    for v6_row in (v6e_live_34[0], v6e_fallback_34[0], v6e_ref_34):
        assert v6_row["single_device_fits"] is False, v6_row
        assert v6_row["usd_per_1m_tokens_model"] is None, v6_row

    # Live v5e (with and without memory_stats) and v5e reference MUST agree (~17.18 decimal GB for 16 GiB)
    v5e_live_17 = build_scorecard_rows(
        {
            "backend": "jax_tpu",
            "device_info": {"device_kind": "TPU v5 lite", "hbm_bytes_limit": 16 * 1024**3},
            "compute": {"peak_bf16_tflops": 179.36},
            "bandwidth": {"peak_triad_gbps": 655.55},
        },
        model_gb=17.0,
    )
    v5e_fallback_17 = build_scorecard_rows(
        {
            "backend": "colab_tpu",
            "colab_meta": {"tpu": "v5e1"},
            "device_info": {"device_kind": "TPU v5 lite", "hbm_bytes_limit": None},
            "compute": {"peak_bf16_tflops": 179.36},
            "bandwidth": {"peak_triad_gbps": 655.55},
        },
        model_gb=17.0,
    )
    v5e_ref_17 = next(r for r in v5e_live_17 if r["target"] == "Cloud TPU v5e-1")
    for v5_row in (v5e_live_17[0], v5e_fallback_17[0], v5e_ref_17):
        assert v5_row["memory_gb"] == 17.18, v5_row
        assert v5_row["single_device_fits"] is True, v5_row
        assert v5_row["usd_per_1m_tokens_model"] is not None, v5_row

    # 3a. Verify colab_tpu scorecard never attaches local TCP comms to a Cloud TPU row
    colab_comms_only = build_scorecard_rows({
        "backend": "colab_tpu",
        "host_platform": "TestHost-arm64",
        "comms": comms,
    })
    measured_comms_only = [r for r in colab_comms_only if r["row_type"] == "measured"]
    assert len(measured_comms_only) == 1
    assert "Local Host Comms" in measured_comms_only[0]["target"]
    assert "Cloud TPU" not in measured_comms_only[0]["target"]
    assert measured_comms_only[0]["memory"] == "host RAM"
    assert measured_comms_only[0]["comms_transport"] == "tcp_loopback"
    assert measured_comms_only[0]["half_rtt_est_us"] == f"{comms['half_rtt_floor_us']} us (half-RTT est.)"

    colab_all_rows = build_scorecard_rows({
        "backend": "colab_tpu",
        "host_platform": "TestHost-arm64",
        "colab_meta": {"tpu": "v5e1"},
        "device_info": {"device_kind": "TPU v5 lite", "hbm_bytes_limit": 16 * 1024**3},
        "compute": {"peak_bf16_tflops": 179.36, "dispatch_floor_us": 125.0},
        "comms": comms,
    })
    measured_all = [r for r in colab_all_rows if r["row_type"] == "measured"]
    assert len(measured_all) == 2
    assert "TPU v5 lite" in measured_all[0]["target"]
    assert measured_all[0]["half_rtt_est_us"] == "—"
    assert "Local Host Comms" in measured_all[1]["target"]
    assert measured_all[1]["half_rtt_est_us"] == f"{comms['half_rtt_floor_us']} us (half-RTT est.)"

    # 3b. Verify CLI rejects invalid economics / timeout args BEFORE running any suite or Colab
    for bad_argv in (
        ["--model-gb", "0", "--backend", "colab_tpu"],
        ["--model-gb", "-2.5", "--backend", "cpu"],
        ["--model-gb", "nan", "--backend", "cpu"],
        ["--model-gb", "inf", "--backend", "cpu"],
        ["--bw-efficiency", "0", "--backend", "cpu"],
        ["--bw-efficiency", "1.5", "--backend", "cpu"],
        ["--batch-size", "0", "--backend", "cpu"],
        ["--colab-timeout", "-1", "--backend", "colab_tpu"],
    ):
        err_buf = StringIO()
        try:
            with redirect_stderr(err_buf):
                main(bad_argv)
        except SystemExit as exc:
            assert exc.code == 2, (bad_argv, exc.code)
        else:  # pragma: no cover
            raise AssertionError(f"Expected SystemExit(2) for {bad_argv}")

    # 3c. Verify bare `--backend colab_tpu --dry-run` prints planned commands to stdout without `--json -`
    dry_buf = StringIO()
    with redirect_stdout(dry_buf):
        rc_dry = main(["--suite", "compute,bandwidth", "--backend", "colab_tpu", "--dry-run"])
    assert rc_dry == 0
    dry_out = dry_buf.getvalue()
    assert "Planned Colab TPU Commands (Dry Run)" in dry_out
    assert "--tpu V6E1" in dry_out and "--tpu V5E1" in dry_out
    assert "--keep" in dry_out and "--emit-markers" in dry_out

    # 4. Verify colab CLI security minimum (>=0.7.6) is enforced BEFORE any authenticated call
    for bad_ver_out, bad_rc in (
        ("Version: 0.7.5\n", 0),
        ("Version: 0.7.6rc1\n", 0),
        ("Version: abcdef1\n", 0),
        ("", 1),
    ):
        ver_calls: List[List[str]] = []

        def _mock_bad_ver_runner(
            cmd: Sequence[str], _out: str = bad_ver_out, _rc: int = bad_rc
        ) -> subprocess.CompletedProcess[str]:
            ver_calls.append(list(cmd))
            if cmd[-1] == "version":
                return subprocess.CompletedProcess(cmd, _rc, _out, "err" if _rc else "")
            raise AssertionError(f"Authenticated command called despite bad colab version: {cmd}")

        try:
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin="colab",
                auth="adc",
                tpu_order=("v5e1",),
                config_path=None,
                timeout_s=120.0,
                runner=_mock_bad_ver_runner,
            )
        except RuntimeError as exc:
            assert "0.7.6" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"Expected RuntimeError for colab version {bad_ver_out!r}")
        assert len(ver_calls) == 1 and ver_calls[0] == ["colab", "version"]

    # 4a. Verify colab_tpu fallback + --keep + endpoint-based teardown verification (version 0.7.6 accepted)
    calls: List[List[str]] = []
    v5e_active = {"live": False, "endpoint": "m-s-v5e1-ok"}

    def _mock_colab_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.7.6\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "Current balance: 100.0 compute units\n", "")
        if "stop" in cmd:
            v5e_active["live"] = False
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
        if "sessions" in cmd:
            # Include an unrelated pre-existing runtime to prove we do not false-positive on it.
            lines = ["[preexisting] m-s-other-99 | Hardware: T4 | Shape: Standard | Variant: GPU"]
            if v5e_active["live"]:
                run_calls = [c for c in calls if "run" in c]
                sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                lines.append(
                    f"[{sname}] {v5e_active['endpoint']} | Hardware: V5E1 | Shape: Standard | Variant: TPU"
                )
            return subprocess.CompletedProcess(cmd, 0, "\n".join(lines) + "\n", "")
        assert "--keep" in cmd
        if "V6E1" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "HTTP 412 REASON_NO_BACKENDS: v6e1 unavailable")
        v5e_active["live"] = True
        payload = {
            "schema": SCHEMA_VERSION,
            "device_info": {"device_kind": "TPU v5 lite", "hbm_bytes_limit": 16 * 1024**3},
            "compute": {
                "status": "measured",
                "backend": "jax_tpu",
                "peak_bf16_tflops": 179.36,
                "peak_int8_tops": 350.69,
                "dispatch_floor_us": 125.0,
                "cases": [],
            },
        }
        out = f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n"
        return subprocess.CompletedProcess(cmd, 0, out, "[colab] Session READY.\n")

    colab_res = run_colab_tpu_suites(
        suites=["compute"],
        quick=True,
        colab_bin="colab",
        auth="adc",
        tpu_order=("v6e1", "v5e1"),
        config_path=None,
        timeout_s=120.0,
        runner=_mock_colab_runner,
    )
    assert colab_res["colab_meta"]["colab_cli_version"] == "0.7.6"
    assert colab_res["colab_meta"]["tpu"] == "v5e1"
    assert colab_res["colab_meta"]["endpoint"] == "m-s-v5e1-ok"
    assert colab_res["colab_meta"]["teardown_verified"] is True
    assert colab_res["colab_meta"]["leaked_endpoints"] == []
    assert colab_res["colab_meta"]["orphaned_endpoints"] == []
    assert colab_res["compute"]["peak_bf16_tflops"] == 179.36
    assert calls[0] == ["colab", "version"]
    assert any("stop" in c for c in calls)
    assert not v5e_active["live"]

    # 4b. Unassign failure where local session record is lost and runtime reappears under `[?]`
    orphan_state = {"stage": "pre", "endpoint": "m-s-leaked-01"}

    def _mock_orphan_leak_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            orphan_state["stage"] = "running"
            payload = {"schema": SCHEMA_VERSION, "compute": {"peak_bf16_tflops": 1.0}}
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n",
                "[colab] Session READY.\n",
            )
        if "stop" in cmd:
            orphan_state["stage"] = "orphaned"
            return subprocess.CompletedProcess(cmd, 1, "", "HTTP 500 unassign failed")
        if "sessions" in cmd:
            if orphan_state["stage"] == "pre":
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            if orphan_state["stage"] == "running":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    f"[?] {orphan_state['endpoint']} | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                    "",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"[?] {orphan_state['endpoint']} | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                "",
            )
        raise AssertionError(f"Unexpected cmd: {cmd}")

    warn_buf = StringIO()
    with redirect_stderr(warn_buf):
        orphan_res = run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v5e1",),
            config_path=None,
            timeout_s=120.0,
            runner=_mock_orphan_leak_runner,
        )
    assert orphan_res["colab_meta"]["teardown_verified"] is False
    assert orphan_res["colab_meta"]["leaked_endpoints"] == []
    assert orphan_res["colab_meta"]["orphaned_endpoints"] == []
    assert orphan_res["colab_meta"]["unattributed_endpoints"] == ["m-s-leaked-01"]
    hint_text = orphan_res["colab_meta"]["recovery_hint"]
    assert "m-s-leaked-01" in hint_text and "confirm ownership before" in hint_text
    assert "retry `" not in hint_text
    assert "teardown unverified" in warn_buf.getvalue()
    orphan_rec = Path(orphan_res["colab_meta"]["recovery_config"])
    assert orphan_rec.parent.is_dir()
    shutil.rmtree(orphan_rec.parent, ignore_errors=True)

    # 4c. Mapped session before stop -> unassign fails -> reappears under `[?]` -> MUST fail verification
    mapped_calls: List[List[str]] = []
    mapped_stage = {"step": "pre", "endpoint": "m-s-mapped-leak"}

    def _mock_mapped_leak_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        mapped_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            mapped_stage["step"] = "post_run"
            payload = {"schema": SCHEMA_VERSION, "compute": {"peak_bf16_tflops": 1.0}}
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n",
                "[colab] Session READY.\n",
            )
        if "stop" in cmd:
            mapped_stage["step"] = "post_stop_orphaned"
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "sessions" in cmd:
            if mapped_stage["step"] == "pre":
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            if mapped_stage["step"] == "post_run":
                run_calls = [c for c in mapped_calls if "run" in c]
                sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    f"[{sname}] {mapped_stage['endpoint']} | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                    "",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"[?] {mapped_stage['endpoint']} | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                "",
            )
        raise AssertionError(f"Unexpected cmd: {cmd}")

    warn_buf2 = StringIO()
    with redirect_stderr(warn_buf2):
        mapped_res = run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v5e1",),
            config_path=None,
            timeout_s=120.0,
            runner=_mock_mapped_leak_runner,
        )
    assert mapped_res["colab_meta"]["endpoint"] == "m-s-mapped-leak"
    assert mapped_res["colab_meta"]["teardown_verified"] is False
    assert mapped_res["colab_meta"]["leaked_endpoints"] == ["m-s-mapped-leak"]
    assert mapped_res["colab_meta"]["orphaned_endpoints"] == ["m-s-mapped-leak"]
    assert "Manage sessions" in mapped_res["colab_meta"]["recovery_hint"]
    mapped_rec = Path(mapped_res["colab_meta"]["recovery_config"])
    assert mapped_rec.parent.is_dir()
    shutil.rmtree(mapped_rec.parent, ignore_errors=True)

    # 4d. Ctrl-C (KeyboardInterrupt) during `colab run --keep` MUST stop the retained runtime
    ki_calls: List[List[str]] = []
    ki_state = {"live": False, "cfg_path": None}

    def _mock_ki_clean_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        ki_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "--config" in cmd:
            ki_state["cfg_path"] = cmd[cmd.index("--config") + 1]
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            ki_state["live"] = True
            Path(str(ki_state["cfg_path"])).write_text("{}", encoding="utf-8")
            raise KeyboardInterrupt()
        if "stop" in cmd:
            ki_state["live"] = False
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
        if "sessions" in cmd:
            if ki_state["live"]:
                run_calls = [c for c in ki_calls if "run" in c]
                sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    f"[{sname}] m-s-ki-clean | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                    "",
                )
            return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
        raise AssertionError(f"Unexpected cmd: {cmd}")

    try:
        run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v5e1",),
            config_path=None,
            timeout_s=120.0,
            runner=_mock_ki_clean_runner,
        )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected KeyboardInterrupt to propagate")
    assert any("stop" in c for c in ki_calls)
    assert ki_state["live"] is False
    assert not Path(str(ki_state["cfg_path"])).parent.exists()

    # 4e. Ctrl-C where teardown fails MUST preserve the temp session config for manual recovery
    ki_fail_state: Dict[str, Any] = {"stage": "pre", "cfg_path": None}

    def _mock_ki_unverified_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "--config" in cmd:
            ki_fail_state["cfg_path"] = cmd[cmd.index("--config") + 1]
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            ki_fail_state["stage"] = "live"
            Path(str(ki_fail_state["cfg_path"])).write_text('{"sess": {"endpoint": "m-s-ki-leak"}}', encoding="utf-8")
            raise KeyboardInterrupt()
        if "stop" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "HTTP 503 unavailable")
        if "sessions" in cmd:
            if ki_fail_state["stage"] == "pre":
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            return subprocess.CompletedProcess(
                cmd,
                0,
                "[?] m-s-ki-leak | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                "",
            )
        raise AssertionError(f"Unexpected cmd: {cmd}")

    ki_warn = StringIO()
    try:
        with redirect_stderr(ki_warn):
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin="colab",
                auth="adc",
                tpu_order=("v5e1",),
                config_path=None,
                timeout_s=120.0,
                runner=_mock_ki_unverified_runner,
            )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected KeyboardInterrupt to propagate")
    preserved_cfg = Path(str(ki_fail_state["cfg_path"]))
    assert preserved_cfg.is_file()
    assert "m-s-ki-leak" in ki_warn.getvalue() and "Manage sessions" in ki_warn.getvalue()
    shutil.rmtree(preserved_cfg.parent, ignore_errors=True)

    # 4f. End-to-end `main()` MUST return nonzero exit status when teardown_verified is False
    e2e_dir = Path(tempfile.mkdtemp(prefix="freetpu-bench-e2e-"))
    try:
        stub_colab = e2e_dir / "fake-colab"
        stub_state = e2e_dir / "state.txt"
        stub_colab.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            f"state = Path({str(stub_state)!r})\n"
            "args = sys.argv[1:]\n"
            "if args and args[-1] == 'version':\n"
            "    print('Version: 0.8.0')\n"
            "elif 'usage' in args:\n"
            "    print('100.0')\n"
            "elif 'run' in args:\n"
            "    state.write_text('live', encoding='utf-8')\n"
            "    payload = {'schema': 'freetpu.bench.v1', 'compute': {'peak_bf16_tflops': 1.0}}\n"
            f"    print({BENCH_JSON_BEGIN!r})\n"
            "    print(json.dumps(payload))\n"
            f"    print({BENCH_JSON_END!r})\n"
            "    print('[colab] Session READY.', file=sys.stderr)\n"
            "elif 'stop' in args:\n"
            "    print('[colab] Session not found.')\n"
            "elif 'sessions' in args:\n"
            "    if state.exists():\n"
            "        print('[?] m-s-orphan-e2e | Hardware: V5E1 | Shape: Standard | Variant: TPU')\n"
            "    else:\n"
            "        print('[colab] No active sessions found on server.')\n",
            encoding="utf-8",
        )
        os.chmod(stub_colab, 0o700)
        e2e_out = StringIO()
        e2e_err = StringIO()
        with redirect_stdout(e2e_out), redirect_stderr(e2e_err):
            rc_leak = main([
                "--suite", "compute",
                "--backend", "colab_tpu",
                "--colab-tpu", "v5e1",
                "--colab-bin", str(stub_colab),
                "--colab-config", str(e2e_dir / "sessions.json"),
                "--json", "-",
            ])
        assert rc_leak == 2, rc_leak
        e2e_report = json.loads(e2e_out.getvalue())
        assert e2e_report["colab_meta"]["teardown_verified"] is False
        assert e2e_report["colab_meta"]["unattributed_endpoints"] == ["m-s-orphan-e2e"]
        assert "m-s-orphan-e2e" in e2e_err.getvalue() and "confirm ownership before" in e2e_err.getvalue()
    finally:
        shutil.rmtree(e2e_dir, ignore_errors=True)

    # 4g. Relative `--colab-config sessions.json` MUST be normalized to an absolute path
    rel_out = StringIO()
    with redirect_stdout(rel_out):
        rc_rel = main([
            "--suite", "compute",
            "--backend", "colab_tpu",
            "--colab-tpu", "v5e1",
            "--colab-config", "sessions.json",
            "--dry-run",
            "--json", "-",
        ])
    assert rc_rel == 0
    rel_cmds = json.loads(rel_out.getvalue())["colab_meta"]["planned_commands"]
    rel_cfg_arg = rel_cmds[0][rel_cmds[0].index("--config") + 1]
    assert Path(rel_cfg_arg).is_absolute() and os.path.dirname(rel_cfg_arg) != ""
    assert rel_cfg_arg == os.path.abspath("sessions.json")

    # 4h. Preflight `colab sessions` failure MUST abort BEFORE `colab run` allocates a TPU
    pre_fail_calls: List[List[str]] = []

    def _mock_pre_sessions_fail_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        pre_fail_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "sessions" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "FileNotFoundError: empty parent path")
        raise AssertionError(f"colab run must not execute after sessions preflight failure: {cmd}")

    try:
        run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v5e1",),
            config_path=Path("sessions.json"),
            timeout_s=120.0,
            runner=_mock_pre_sessions_fail_runner,
        )
    except RuntimeError as exc:
        assert "sessions preflight failed" in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError on failed sessions preflight")
    assert not any("run" in c for c in pre_fail_calls)
    sess_cfg = pre_fail_calls[-1][pre_fail_calls[-1].index("--config") + 1]
    assert Path(sess_cfg).is_absolute()

    # 4i. Unrelated concurrent `[?]` session MUST NOT fail teardown when this run's named endpoint disappeared
    conc_calls: List[List[str]] = []
    conc_state = {"stage": "pre", "ours": "m-s-ours-clean", "other": "m-s-concurrent-browser"}

    def _mock_concurrent_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        conc_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            conc_state["stage"] = "post_run"
            payload = {"schema": SCHEMA_VERSION, "compute": {"peak_bf16_tflops": 1.0}}
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n",
                "[colab] Session READY.\n",
            )
        if "stop" in cmd:
            conc_state["stage"] = "post_stop"
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
        if "sessions" in cmd:
            if conc_state["stage"] == "pre":
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            if conc_state["stage"] == "post_run":
                run_calls = [c for c in conc_calls if "run" in c]
                sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    (
                        f"[{sname}] {conc_state['ours']} | Hardware: V5E1 | Shape: Standard | Variant: TPU\n"
                        f"[?] {conc_state['other']} | Hardware: T4 | Shape: Standard | Variant: GPU\n"
                    ),
                    "",
                )
            # After stop: our endpoint is GONE, concurrent unrelated [?] session remains
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"[?] {conc_state['other']} | Hardware: T4 | Shape: Standard | Variant: GPU\n",
                "",
            )
        raise AssertionError(f"Unexpected cmd: {cmd}")

    conc_err = StringIO()
    with redirect_stderr(conc_err):
        conc_res = run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v5e1",),
            config_path=None,
            timeout_s=120.0,
            runner=_mock_concurrent_runner,
        )
    assert conc_res["colab_meta"]["endpoint"] == "m-s-ours-clean"
    assert conc_res["colab_meta"]["teardown_verified"] is True
    assert conc_res["colab_meta"]["leaked_endpoints"] == []
    assert conc_res["colab_meta"]["orphaned_endpoints"] == []
    assert conc_res["colab_meta"]["unattributed_endpoints"] == ["m-s-concurrent-browser"]
    assert "unattributed concurrent" in conc_err.getvalue()

    # 4j. SIGTERM during `colab run --keep` MUST run teardown (and defer a second SIGTERM during stop)
    if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
        sig_calls: List[List[str]] = []
        sig_state = {"live": False, "second_sent": False}
        orig_handler = signal.getsignal(signal.SIGTERM)

        def _mock_sigterm_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
            sig_calls.append(list(cmd))
            if cmd[-1] == "version":
                return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
            if "usage" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
            if "run" in cmd:
                sig_state["live"] = True
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("SIGTERM should have interrupted runner")
            if "stop" in cmd:
                if not sig_state["second_sent"]:
                    sig_state["second_sent"] = True
                    # A second SIGTERM during teardown MUST NOT abort cleanup mid-flight
                    os.kill(os.getpid(), signal.SIGTERM)
                sig_state["live"] = False
                return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
            if "sessions" in cmd:
                if sig_state["live"]:
                    run_calls = [c for c in sig_calls if "run" in c]
                    sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        f"[{sname}] m-s-sigterm-clean | Hardware: V5E1 | Shape: Standard | Variant: TPU\n",
                        "",
                    )
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            raise AssertionError(f"Unexpected cmd: {cmd}")

        try:
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin="colab",
                auth="adc",
                tpu_order=("v5e1",),
                config_path=None,
                timeout_s=120.0,
                runner=_mock_sigterm_runner,
            )
        except _ColabSigtermInterrupt:
            pass
        else:  # pragma: no cover
            raise AssertionError("Expected _ColabSigtermInterrupt to propagate")
        assert any("stop" in c for c in sig_calls)
        assert sig_state["second_sent"] is True
        assert sig_state["live"] is False
        assert signal.getsignal(signal.SIGTERM) == orig_handler

    # 4k. Refused v6e allocation while a concurrent `[?]` runtime appears MUST keep it unattributed and fall back to v5e
    ref_calls: List[List[str]] = []
    ref_state = {"v6e_tried": False, "v5e_live": False}

    def _mock_refused_with_concurrent_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        ref_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "stop" in cmd:
            ref_state["v5e_live"] = False
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
        if "sessions" in cmd:
            lines: List[str] = []
            if ref_state["v6e_tried"]:
                # Another workload started concurrently under the same account -> `[?]`
                lines.append("[?] m-s-other-workload | Hardware: T4 | Shape: Standard | Variant: GPU")
            if ref_state["v5e_live"]:
                run_calls = [c for c in ref_calls if "run" in c]
                sname = run_calls[-1][run_calls[-1].index("-s") + 1]
                lines.append(f"[{sname}] m-s-v5e-fallback | Hardware: V5E1 | Shape: Standard | Variant: TPU")
            out_s = ("\n".join(lines) + "\n") if lines else "[colab] No active sessions found on server.\n"
            return subprocess.CompletedProcess(cmd, 0, out_s, "")
        if "V6E1" in cmd:
            ref_state["v6e_tried"] = True
            return subprocess.CompletedProcess(cmd, 1, "", "HTTP 412 REASON_NO_BACKENDS: v6e1 unavailable")
        ref_state["v5e_live"] = True
        payload = {"schema": SCHEMA_VERSION, "compute": {"peak_bf16_tflops": 179.36}}
        return subprocess.CompletedProcess(
            cmd,
            0,
            f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n",
            "[colab] Session READY.\n",
        )

    ref_err = StringIO()
    with redirect_stderr(ref_err):
        ref_res = run_colab_tpu_suites(
            suites=["compute"],
            quick=True,
            colab_bin="colab",
            auth="adc",
            tpu_order=("v6e1", "v5e1"),
            config_path=None,
            timeout_s=120.0,
            runner=_mock_refused_with_concurrent_runner,
        )
    assert ref_res["colab_meta"]["tpu"] == "v5e1"
    assert ref_res["colab_meta"]["teardown_verified"] is True
    assert ref_res["colab_meta"]["leaked_endpoints"] == []
    assert ref_res["colab_meta"]["attempts"][0]["unattributed_endpoints"] == ["m-s-other-workload"]
    assert "terminate" not in ref_err.getvalue().lower() or "not terminated" in ref_err.getvalue().lower()

    # 4l. Non-interactive `_default_colab_runner` MUST pass stdin=DEVNULL so OAuth prompts cannot hang
    oauth_dir = Path(tempfile.mkdtemp(prefix="freetpu-bench-oauth-"))
    try:
        oauth_stub = oauth_dir / "fake-colab-oauth"
        oauth_stub.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "args = sys.argv[1:]\n"
            "if args and args[-1] == 'version':\n"
            "    print('Version: 0.8.0')\n"
            "elif 'usage' in args:\n"
            "    print('Open https://accounts.google.com/... and enter code:')\n"
            "    input('Enter verification code: ')\n",
            encoding="utf-8",
        )
        os.chmod(oauth_stub, 0o700)
        t_oauth = time.monotonic()
        try:
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin=str(oauth_stub),
                auth="oauth2",
                tpu_order=("v5e1",),
                config_path=oauth_dir / "sessions.json",
                timeout_s=30.0,
                runner=_default_colab_runner,
            )
        except RuntimeError as exc:
            assert "interactively" in str(exc) and "oauth2" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError when OAuth prompt hits EOF on DEVNULL stdin")
        assert time.monotonic() - t_oauth < 5.0
    finally:
        shutil.rmtree(oauth_dir, ignore_errors=True)

    # 4m. Ambiguous allocation failure (ReadTimeout / HTTP 503) MUST keep teardown_verified=False,
    # block v5e fallback, preserve temp state, and surface recovery guidance BOTH when a new `[?]`
    # entry appears immediately AND when the first `colab sessions` check is still empty!
    for post_sessions_mode in ("new_unattributed", "empty"):
        for ambig_err_msg in ("ReadTimeout: read timed out waiting for assign", "HTTP 503 Service Unavailable"):
            ambig_calls: List[List[str]] = []
            ambig_state: Dict[str, Any] = {"tried": False, "cfg": None}

            def _mock_ambig_runner(
                cmd: Sequence[str],
                _msg: str = ambig_err_msg,
                _mode: str = post_sessions_mode,
            ) -> subprocess.CompletedProcess[str]:
                ambig_calls.append(list(cmd))
                if "--config" in cmd:
                    ambig_state["cfg"] = cmd[cmd.index("--config") + 1]
                if cmd[-1] == "version":
                    return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
                if "usage" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
                if "sessions" in cmd:
                    if ambig_state["tried"] and _mode == "new_unattributed":
                        return subprocess.CompletedProcess(
                            cmd,
                            0,
                            "[?] m-s-ambig-timeout | Hardware: V6E1 | Shape: Standard | Variant: TPU\n",
                            "",
                        )
                    return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
                if "V6E1" in cmd:
                    ambig_state["tried"] = True
                    return subprocess.CompletedProcess(cmd, 1, "", _msg)
                raise AssertionError(f"Must NOT fall back to v5e1 after ambiguous failure: {cmd}")

            ambig_err = StringIO()
            try:
                with redirect_stderr(ambig_err):
                    run_colab_tpu_suites(
                        suites=["compute"],
                        quick=True,
                        colab_bin="colab",
                        auth="adc",
                        tpu_order=("v6e1", "v5e1"),
                        config_path=None,
                        timeout_s=120.0,
                        runner=_mock_ambig_runner,
                    )
            except RuntimeError as exc:
                assert "teardown_verified=False" in str(exc), str(exc)
            else:  # pragma: no cover
                raise AssertionError("Expected RuntimeError after ambiguous allocation failure")
            assert not any("V5E1" in c for c in ambig_calls)
            err_txt = ambig_err.getvalue()
            if post_sessions_mode == "new_unattributed":
                assert "m-s-ambig-timeout" in err_txt and "confirm ownership before" in err_txt, err_txt
            else:
                assert "still be provisioning" in err_txt and "Re-check" in err_txt, err_txt
            ambig_cfg = Path(str(ambig_state["cfg"]))
            assert ambig_cfg.parent.is_dir()
            shutil.rmtree(ambig_cfg.parent, ignore_errors=True)

    # 4m2. Interrupt during allocation with empty `colab sessions` MUST keep teardown_verified=False
    ki_empty_state: Dict[str, Any] = {"cfg": None}

    def _mock_ki_empty_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if "--config" in cmd:
            ki_empty_state["cfg"] = cmd[cmd.index("--config") + 1]
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            raise KeyboardInterrupt()
        if "stop" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session not found.\n", "")
        if "sessions" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
        raise AssertionError(f"Unexpected cmd: {cmd}")

    ki_empty_err = StringIO()
    try:
        with redirect_stderr(ki_empty_err):
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin="colab",
                auth="adc",
                tpu_order=("v6e1",),
                config_path=None,
                timeout_s=120.0,
                runner=_mock_ki_empty_runner,
            )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected KeyboardInterrupt to propagate")
    assert "still be provisioning" in ki_empty_err.getvalue()
    ki_empty_cfg = Path(str(ki_empty_state["cfg"]))
    assert ki_empty_cfg.parent.is_dir()
    shutil.rmtree(ki_empty_cfg.parent, ignore_errors=True)

    # 4n. Hung `colab run` (e.g. kernel init stall) MUST hit host-side deadline and run teardown
    hang_dir = Path(tempfile.mkdtemp(prefix="freetpu-bench-hang-"))
    try:
        hang_stub = hang_dir / "fake-colab-hang"
        hang_state = hang_dir / "state.txt"
        hang_stopped = hang_dir / "stopped.txt"
        hang_stub.write_text(
            f"#!{sys.executable}\n"
            "import sys, time\n"
            "from pathlib import Path\n"
            f"state = Path({str(hang_state)!r})\n"
            f"stopped = Path({str(hang_stopped)!r})\n"
            "args = sys.argv[1:]\n"
            "if args and args[-1] == 'version':\n"
            "    print('Version: 0.8.0')\n"
            "elif 'usage' in args:\n"
            "    print('100.0')\n"
            "elif 'run' in args:\n"
            "    sname = args[args.index('-s') + 1]\n"
            "    state.write_text(sname, encoding='utf-8')\n"
            "    time.sleep(15.0)\n"
            "elif 'stop' in args:\n"
            "    stopped.write_text('stopped', encoding='utf-8')\n"
            "    if state.exists():\n"
            "        state.unlink()\n"
            "    print('[colab] Session terminated.')\n"
            "elif 'sessions' in args:\n"
            "    if state.exists():\n"
            "        sname = state.read_text(encoding='utf-8').strip()\n"
            "        print(f'[{sname}] m-s-hung-kernel | Hardware: V6E1 | Shape: Standard | Variant: TPU')\n"
            "    else:\n"
            "        print('[colab] No active sessions found on server.')\n",
            encoding="utf-8",
        )
        os.chmod(hang_stub, 0o700)
        t_hang = time.monotonic()
        try:
            run_colab_tpu_suites(
                suites=["compute"],
                quick=True,
                colab_bin=str(hang_stub),
                auth="adc",
                tpu_order=("v6e1", "v5e1"),
                config_path=hang_dir / "sessions.json",
                timeout_s=0.15,
                runner=_make_default_colab_runner(run_timeout_s=0.15, provisioning_grace_s=0.05, control_timeout_s=5.0),
            )
        except RuntimeError as exc:
            assert "host-side deadline" in str(exc), str(exc)
            assert "teardown_verified=True" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError when colab run exceeds host-side deadline")
        assert time.monotonic() - t_hang < 3.0
        assert hang_stopped.is_file()
        assert not hang_state.exists()
    finally:
        shutil.rmtree(hang_dir, ignore_errors=True)

    # 4o. Timeout on initial post-run `colab sessions` lookup MUST still attempt `colab stop -s` + recheck
    to_calls: List[List[str]] = []
    to_state = {"sessions_calls": 0, "stopped": False}

    def _mock_sessions_timeout_runner(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        to_calls.append(list(cmd))
        if cmd[-1] == "version":
            return subprocess.CompletedProcess(cmd, 0, "Version: 0.8.0\n", "")
        if "usage" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "100.0\n", "")
        if "run" in cmd:
            payload = {"schema": SCHEMA_VERSION, "compute": {"peak_bf16_tflops": 179.36}}
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"{BENCH_JSON_BEGIN}\n{json.dumps(payload)}\n{BENCH_JSON_END}\n",
                "[colab] Session READY.\n",
            )
        if "stop" in cmd:
            to_state["stopped"] = True
            return subprocess.CompletedProcess(cmd, 0, "[colab] Session terminated.\n", "")
        if "sessions" in cmd:
            to_state["sessions_calls"] += 1
            if to_state["sessions_calls"] == 1:
                # Preflight sessions check succeeds
                return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
            if to_state["sessions_calls"] == 2:
                # First post-run sessions lookup times out!
                raise subprocess.TimeoutExpired(cmd=list(cmd), timeout=90.0)
            # Post-stop recheck confirms server has zero active sessions
            return subprocess.CompletedProcess(cmd, 0, "[colab] No active sessions found on server.\n", "")
        raise AssertionError(f"Unexpected cmd: {cmd}")

    to_res = run_colab_tpu_suites(
        suites=["compute"],
        quick=True,
        colab_bin="colab",
        auth="adc",
        tpu_order=("v5e1",),
        config_path=None,
        timeout_s=120.0,
        runner=_mock_sessions_timeout_runner,
    )
    assert to_state["stopped"] is True
    assert any("stop" in c for c in to_calls)
    assert to_state["sessions_calls"] == 3
    assert to_res["colab_meta"]["teardown_verified"] is True

    # 4p. `_default_colab_runner` MUST force COLAB_CLI_ASSIGN_RETRIES=0 (overriding ambient env)
    # and a retried timeout + HTTP 412 refusal MUST NOT be treated as a definitive refusal
    prev_retries = os.environ.get("COLAB_CLI_ASSIGN_RETRIES")
    os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "3"
    try:
        env_probe = _default_colab_runner(
            [sys.executable, "-c", "import os; print(os.environ.get('COLAB_CLI_ASSIGN_RETRIES', ''))"],
            timeout_s=10.0,
        )
        assert env_probe.returncode == 0
        assert env_probe.stdout.strip() == "0", repr(env_probe.stdout)
    finally:
        if prev_retries is None:
            os.environ.pop("COLAB_CLI_ASSIGN_RETRIES", None)
        else:
            os.environ["COLAB_CLI_ASSIGN_RETRIES"] = prev_retries
    assert is_colab_allocation_error(1, "HTTP 412 Allocation refused: REASON_NO_BACKENDS") is True
    assert (
        is_colab_allocation_error(
            1,
            "ReadTimeout: timed out waiting for assign; retrying...\nHTTP 412 Allocation refused",
        )
        is False
    )

    # 4q. `--suite comms --backend jax_tpu` MUST skip JAX/TPU initialization completely
    had_jax = "jax" in sys.modules
    prev_jax = sys.modules.get("jax")
    sys.modules["jax"] = None  # type: ignore[assignment]
    try:
        jax_comms_out = StringIO()
        with redirect_stdout(jax_comms_out):
            rc_jax_comms = main(["--suite", "comms", "--backend", "jax_tpu", "--quick", "--json", "-"])
        assert rc_jax_comms == 0, rc_jax_comms
        jax_comms_rep = json.loads(jax_comms_out.getvalue())
        assert jax_comms_rep["backend"] == "jax_tpu"
        assert "comms" in jax_comms_rep and jax_comms_rep["comms"]["status"] == "measured"
        assert "device_info" not in jax_comms_rep
        assert "compute" not in jax_comms_rep and "bandwidth" not in jax_comms_rep
        measured_jax_comms = [r for r in jax_comms_rep["scorecard"] if r["row_type"] == "measured"]
        assert len(measured_jax_comms) == 1 and "Local Host Comms" in measured_jax_comms[0]["target"]
    finally:
        if had_jax:
            sys.modules["jax"] = prev_jax  # type: ignore[assignment]
        else:
            sys.modules.pop("jax", None)

    # 5. Verify this script compiles and loads under a `colab run` executable prelude
    src_text = Path(__file__).read_text(encoding="utf-8")
    compile("import os\nos.environ['X'] = '1'\n" + src_text, "colab_cell", "exec")
    compiled_cell = compile("_colab_prelude = 1\n" + src_text, "colab_cell", "exec")
    cell_ns: Dict[str, Any] = {"__name__": "colab_cell_test"}
    exec(compiled_cell, cell_ns)
    assert callable(cell_ns.get("run_jax_tpu_suites"))
    assert callable(cell_ns.get("main"))

    print(
        json.dumps(
            {
                "status": "ok",
                "self_test": "passed",
                "suites_checked": list(VALID_SUITES),
                "baselines_checked": len(PUBLISHED_BASELINES),
                "colab_fallback_verified": True,
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="freetpu-bench",
        description="Unified cross-TPU benchmark suite (compute, bandwidth, comms) + scorecard.",
    )
    p.add_argument("--suite", default="all", help="Comma-separated suites: compute,bandwidth,comms,all")
    p.add_argument("--backend", choices=VALID_BACKENDS, default="cpu", help="Execution backend")
    p.add_argument("--json", dest="json_out", default=None, help="Output JSON path (or '-' for stdout)")
    p.add_argument("--markdown", dest="md_out", default=None, help="Output Markdown scorecard path (or '-' for stdout)")
    p.add_argument("--emit-markers", action="store_true", help="Print JSON payload wrapped in FREETPU_BENCH markers")
    p.add_argument("--quick", action="store_true", help="Run smaller shapes/iterations for fast smoke validation")
    p.add_argument("--self-test", action="store_true", help="Run offline deterministic self-tests and exit")
    p.add_argument("--model-gb", type=float, default=8.0, help="Weight footprint (GB) for $/1M-token decode model")
    p.add_argument("--bw-efficiency", type=float, default=0.70, help="Memory BW efficiency factor (0..1]")
    p.add_argument("--batch-size", type=int, default=1, help="Decode batch size for $/1M-token projection")
    p.add_argument("--comms-host", default="127.0.0.1", help="Target host for comms suite")
    p.add_argument("--comms-port", type=int, default=0, help="Target port for comms suite (0 = local loopback)")
    p.add_argument("--comms-server", action="store_true", help="Start TCP activation-hop server and block")
    p.add_argument("--colab-bin", default=None, help="Path to google-colab-cli (>=0.7.6) `colab` binary")
    p.add_argument("--colab-auth", default="adc", choices=("adc", "oauth2"), help="Colab CLI auth strategy")
    p.add_argument("--colab-tpu", default="auto", choices=("auto", "v6e1", "v5e1"), help="Colab TPU target")
    p.add_argument("--colab-config", default=None, help="Isolated `--config` path for Colab session state")
    p.add_argument("--colab-timeout", type=float, default=300.0, help="Colab execution timeout (seconds)")
    p.add_argument("--allow-jax-cpu", action="store_true", help="Allow jax_tpu backend to run on JAX CPU for testing")
    p.add_argument("--dry-run", action="store_true", help="With --backend colab_tpu, print planned commands only")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not (0 <= args.comms_port <= 65535):
        parser.error(f"--comms-port must be in 0..65535 (got {args.comms_port!r})")

    if args.comms_server:
        run_comms_server(args.comms_host, args.comms_port)
        return 0

    try:
        validate_economics_params(args.model_gb, args.bw_efficiency, args.batch_size)
        suites = parse_suites(args.suite)
    except ValueError as exc:
        parser.error(str(exc))

    if not math.isfinite(args.colab_timeout) or args.colab_timeout < 0.0:
        parser.error(f"--colab-timeout must be finite and >= 0 (got {args.colab_timeout!r})")
    report: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": args.backend,
        "host_platform": f"{platform.system()}-{platform.machine()}",
        "suites": suites,
        "economics_assumptions": {
            "model_gb": args.model_gb,
            "bw_efficiency": args.bw_efficiency,
            "batch_size": args.batch_size,
        },
    }

    if args.backend in ("cpu", "buenos_mock"):
        if "compute" in suites:
            report["compute"] = run_compute_local(args.backend, quick=args.quick)
        if "bandwidth" in suites:
            report["bandwidth"] = run_bandwidth_local(args.backend, quick=args.quick)
        if "comms" in suites:
            report["comms"] = run_comms_suite(args.comms_host, args.comms_port, quick=args.quick)
    elif args.backend == "jax_tpu":
        if "compute" in suites or "bandwidth" in suites:
            jax_res = run_jax_tpu_suites(suites, quick=args.quick, allow_jax_cpu=args.allow_jax_cpu)
            report.update(jax_res)
        if "comms" in suites:
            report["comms"] = run_comms_suite(args.comms_host, args.comms_port, quick=args.quick)
    elif args.backend == "colab_tpu":
        tpu_order = ("v6e1", "v5e1") if args.colab_tpu == "auto" else (args.colab_tpu,)
        cfg_path = Path(os.path.abspath(os.path.expanduser(args.colab_config))) if args.colab_config else None
        colab_res = run_colab_tpu_suites(
            suites=suites,
            quick=args.quick,
            colab_bin=resolve_colab_bin(args.colab_bin),
            auth=args.colab_auth,
            tpu_order=tpu_order,
            config_path=cfg_path,
            timeout_s=args.colab_timeout,
            dry_run=args.dry_run,
        )
        for key in ("device_info", "compute", "bandwidth", "colab_meta"):
            if key in colab_res:
                report[key] = colab_res[key]
        if "host_platform" in colab_res and "colab_meta" in report:
            report["colab_meta"]["remote_host_platform"] = colab_res["host_platform"]
        if "comms" in suites and not args.dry_run:
            report["comms"] = run_comms_suite(args.comms_host, args.comms_port, quick=args.quick)

    report["scorecard"] = build_scorecard_rows(
        report,
        model_gb=args.model_gb,
        bw_efficiency=args.bw_efficiency,
        batch_size=args.batch_size,
    )

    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    md_text = render_markdown(report)

    if args.emit_markers:
        print(BENCH_JSON_BEGIN)
        print(json_text, end="")
        print(BENCH_JSON_END)

    if args.json_out == "-":
        sys.stdout.write(json_text)
    elif args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json_text, encoding="utf-8")

    if args.md_out == "-":
        sys.stdout.write(md_text)
    elif args.md_out:
        out_p = Path(args.md_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_text, encoding="utf-8")

    if not args.emit_markers and not args.json_out and not args.md_out:
        sys.stdout.write(md_text)

    colab_meta = report.get("colab_meta")
    if isinstance(colab_meta, dict) and colab_meta.get("teardown_verified") is False:
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except _ColabSigtermInterrupt:
        raise SystemExit(143)
